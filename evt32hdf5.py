import rosbag
from event_camera_py import Decoder
import h5py
import numpy as np
from tqdm import tqdm

class Event:
    def __init__(self, h5file, topic, bag):
        self.h5file = h5file
        self.topic = topic
        self.bag = bag

        if topic in h5file:
            self.group = h5file[topic]
        else:
            self.group = h5file.create_group(topic)

        event_count = self.get_event_count(bag, topic)
        # print(event_count)
        self.chunk_size = 40000

        self.x_ds = self.group.create_dataset(
            "x",
            (event_count,),
            compression="lzf",
            dtype="u2",
            chunks=(self.chunk_size,),
            maxshape=(None,),
        )
        self.y_ds = self.group.create_dataset(
            "y",
            (event_count,),
            compression="lzf",
            dtype="u2",
            chunks=(self.chunk_size,),
            maxshape=(None,),
        )
        self.t_ds = self.group.create_dataset(
            "t",
            (event_count,),
            compression="lzf",
            dtype="i8",
            chunks=(self.chunk_size,),
            maxshape=(None,),
        )
        self.p_ds = self.group.create_dataset(
            "p",
            (event_count,),
            compression="lzf",
            dtype="i1",
            chunks=(self.chunk_size,),
            maxshape=(None,),
        )

        # 创建四个列表和一个计数器 缓存区
        self.x_buf = []  
        self.y_buf = []  
        self.t_buf = []  
        self.p_buf = []  
        self.count_buf = 0

        # 写入位置
        self.start = 0
        self.prev_msg = None

        # 创建索引映射
        self.has_index_map = False

        self.decoder = Decoder()

    # 统计给定 ROS bag 文件中指定话题的事件相机数据总数
    def get_event_count(self, bag, topic):
        count = 0
        d = Decoder()
        # 获取话题的消息数量，用于进度条
        total_messages = bag.get_message_count(topic_filters=[topic])
        # 使用进度条
        with tqdm(total=total_messages, desc="Counting events", unit="msg") as pbar:
            for topic, msg, t in bag.read_messages(topics=[topic]):
                d.decode_bytes(
                    msg.encoding, msg.width, msg.height, msg.time_base, msg.events
                )
                cd_events = d.get_cd_events()
                count += cd_events.shape[0]
                pbar.update(1)  # 更新进度条
        return count
    
    # 把数据添加到缓冲区
    def add_events(self, x, y, t, p):
        msg_shape = x.shape[0]

        if msg_shape == 0:
            return

        self.x_buf.append(x)
        self.y_buf.append(y)
        self.t_buf.append(t)
        self.p_buf.append(p)
        self.count_buf += msg_shape

    # 将缓冲区写入hdf5
    def flush_buffer(self, ignore_chunk=False):
        x = np.concatenate(self.x_buf)
        y = np.concatenate(self.y_buf)
        t = np.concatenate(self.t_buf)
        p = np.concatenate(self.p_buf)

        if not ignore_chunk:
            added_idx = self.chunk_size * (x.shape[0] // self.chunk_size)
        else:
            added_idx = x.shape[0]

        self.end_idx = self.start + added_idx

        self.x_ds[self.start : self.end_idx] = x[:added_idx]
        self.y_ds[self.start : self.end_idx] = y[:added_idx]
        self.t_ds[self.start : self.end_idx] = t[:added_idx]
        self.p_ds[self.start : self.end_idx] = p[:added_idx]

        self.start = self.start + added_idx

        if not ignore_chunk:
            self.x_buf = [x[added_idx:]]
            self.y_buf = [y[added_idx:]]
            self.t_buf = [t[added_idx:]]
            self.p_buf = [p[added_idx:]]
            self.count_buf = self.count_buf - added_idx
        else:
            self.x_buf = []
            self.y_buf = []
            self.t_buf = []
            self.p_buf = []
            self.count_buf = 0

    # 处理来自 ROS bag 的事件相机数据消息，并将其转换为格式化的数据存储在 HDF5 文件中
    def process(self, msg, topic=None, bag_time=None):
        self.decoder.decode_bytes(
            msg.encoding, msg.width, msg.height, msg.time_base, msg.events
        )
        cd_events = self.decoder.get_cd_events()
        trig_events = self.decoder.get_ext_trig_events()

        t = cd_events['t']

        has_started = t >= 0

        x = cd_events['x'][has_started]
        y = cd_events['y'][has_started]
        t = t[has_started]
        p = cd_events['p'][has_started]

        self.add_events(x, y, t, p)

        if self.count_buf > self.chunk_size:
            self.flush_buffer()

    # 确保所有剩余数据都被处理和保存
    def finish(self):
        # 即使缓冲区中的数据量没有达到 chunk_size 也进行写入，确保了所有数据都被保存
        self.flush_buffer(ignore_chunk=True)
        self.x_ds.resize( (self.end_idx,) )
        self.y_ds.resize( (self.end_idx,) )
        self.t_ds.resize( (self.end_idx,) )
        self.p_ds.resize( (self.end_idx,) )

        self.compute_ms_index_map()

    def primary_time_ds(self):
        raise NotImplementedError("This would lead to something very expensive being computed")

    # 用来生成时间戳索引映射
    def index_map(self, index_map_ds, time_ds):
        index_map_ds_cl = 0

        remaining_times = time_ds[...].copy()
        cur_loc = 0
        chunk_size = 10000000
        num_events = self.t_ds.shape[0]

        while remaining_times.shape[0] > 0 and cur_loc < num_events:
            end = min( num_events, cur_loc+chunk_size )
            idx = cur_loc + np.searchsorted(self.t_ds[cur_loc:end], remaining_times)

            idx_remaining = (idx == end)
            idx_found = (idx < end)

            index_map_ds[index_map_ds_cl:index_map_ds_cl+idx_found.sum()] = idx[idx_found]

            remaining_times = remaining_times[idx_remaining]
            cur_loc = cur_loc + chunk_size
            index_map_ds_cl += idx_found.sum()

    # 生成一个索引映射将毫秒级时间戳与事件数据集中的索引关联起来
    def compute_ms_index_map(self):
        time_ds = np.arange(0, int(np.floor(self.t_ds[-1] / 1e3)) ) * 1e3
        index_map_ds = self.group.create_dataset("ms_map_idx", time_ds.shape, dtype="u8")

        self.index_map( index_map_ds, time_ds )

def process_events_for_topic(h5file, topic, bag):
    event_processor = Event(h5file, topic, bag)
    total_messages = bag.get_message_count(topic_filters=[topic])

    with tqdm(total=total_messages, desc=f"Processing {topic}", unit="msg") as pbar:
        for topic, msg, t in bag.read_messages(topics=[event_processor.topic]):
            event_processor.process(msg)
            pbar.update(1)  

    event_processor.finish()    

def process_bag(bag_file, h5_file):
    bag = rosbag.Bag(bag_file)
    h5file = h5py.File(h5_file, "w")

    process_events_for_topic(h5file, "/event_cam_left/events", bag)
    process_events_for_topic(h5file, "/event_cam_right/events", bag)

    h5file.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", help="ROS bag file path")
    parser.add_argument("--h5", help="Output HDF5 file path")
    args = parser.parse_args()
    process_bag(args.bag, args.h5)

    