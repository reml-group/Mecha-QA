# -*- coding: utf-8 -*-
from nebula2.gclient.net import ConnectionPool
from nebula2.Config import Config
import ctypes
import time

# coding
import sys
import codecs
import os

# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


space_dict = {1: "mechanical_kg"}


def bytes_to_long(bytes):
    assert len(bytes) == 8
    return sum((b << (k * 8) for k, b in enumerate(bytes)))


def murmur64(data, seed=0xC70F6907):
    m = ctypes.c_uint64(0xC6A4A7935BD1E995).value
    r = ctypes.c_uint32(47).value
    MASK = ctypes.c_uint64(2**64 - 1).value
    data_as_bytes = bytearray(data)
    seed = ctypes.c_uint64(seed).value
    h = seed ^ ((m * len(data_as_bytes)) & MASK)
    off = int(len(data_as_bytes) / 8) * 8
    for ll in range(0, off, 8):
        k = bytes_to_long(data_as_bytes[ll : ll + 8])
        k = (k * m) & MASK
        k = k ^ ((k >> r) & MASK)
        k = (k * m) & MASK
        h = h ^ k
        h = (h * m) & MASK
    l = len(data_as_bytes) & 7
    if l >= 7:
        h = h ^ (data_as_bytes[off + 6] << 48)
    if l >= 6:
        h = h ^ (data_as_bytes[off + 5] << 40)
    if l >= 5:
        h = h ^ (data_as_bytes[off + 4] << 32)
    if l >= 4:
        h = h ^ (data_as_bytes[off + 3] << 24)
    if l >= 3:
        h = h ^ (data_as_bytes[off + 2] << 16)
    if l >= 2:
        h = h ^ (data_as_bytes[off + 1] << 8)
    if l >= 1:
        h = h ^ data_as_bytes[off]
        h = (h * m) & MASK
    h = h ^ ((h >> r) & MASK)
    h = (h * m) & MASK
    h = h ^ ((h >> r) & MASK)
    return ctypes.c_long(h).value


def parse_Res(Resultset):
    """
    Nebula返回一个Resultset对象
    实现Resultset->list
    """
    res = []
    try:
        nums = Resultset.row_size()
        for i in range(nums):
            res.append(Resultset.row_values(i))
        return res
    except:
        return []


class nebula_database(object):
    def __init__(self, account="root", password="nebula", ip="219.245.186.43", port=9669, max_try=500):
        # 定义配置,port=50004是知云系统，9669是增材
        self.account = account
        self.password = password
        self.ip = ip
        self.port = port
        config = Config()
        config.max_connection_pool_size = 10
        # 初始化连接池
        self.connection_pool = ConnectionPool()
        # 如果给定的服务器正常，则返回true，否则返回false。
        while max_try:
            try:
                self.connection_pool.init([(ip, port)], config)
                break
            except:
                max_try -= 1
                pass
        # assert self.connection_pool.init([(ip, port)], config), "connect error"
        self.session = self.connection_pool.get_session(self.account, self.password)

    def get_jobid(self):
        res = parse_Res(self.session.execute("SUBMIT JOB STATS"))
        if len(res) > 0:
            return str(res[0][0])
        return ""

    def create_new_space(
        self, sapce_name, partition_num=5, replica_factor=1, vid_type="INT64"
    ):
        # self.session = self.connection_pool.get_session(self.account, self.password)
        nGQL = f"CREATE SPACE {sapce_name} (partition_num={partition_num},replica_factor={replica_factor},vid_type={vid_type})"
        self.session.execute(nGQL)
        # print(self.session.execute("show spaces"))
        # time.sleep(5)
        # self.session.execute(f"USE {sapce_name}")
        # time.sleep(5)
        while self.get_jobid() == "":
            self.session.execute(f"USE {sapce_name}")
            time.sleep(0.1)
        tag_nGQL = f"CREATE TAG entity(name string)"
        edge_nGQL = f"CREATE EDGE relation(name string)"
        self.session.execute(tag_nGQL)
        self.session.execute(edge_nGQL)
        temp = self.get_jobid()
        while True:
            job_id = self.get_jobid()
            time.sleep(0.3)
            new_id = self.get_jobid()
            if new_id != job_id:
                break
            if temp == job_id:
                self.session.execute(f"STOP JOB {job_id}")
                temp = str(int(job_id) + 1)
            else:
                break
            time.sleep(0.1)

    def use_space(self, space_id):
        # self.session = self.connection_pool.get_session(self.account, self.password)
        # 选择图空间
        self.session.execute(f"USE {space_dict[space_id]}")

    def submit(self, space_id):
        self.use_space(space_id)
        job_id = self.session.execute("SUBMIT JOB STATS")
        time.sleep(1)

    def get_all_neigh(self, entity_name, relation):
        vid = str(murmur64(bytes(entity_name, encoding="utf8"), seed=0xC70F6907))
        # vid = 1592061087
        # nGQL = f"MATCH (v:entity)-[e:relation]-(v2) WHERE id(v) == vid RETURN v.name AS s_Name, v2.name AS d_Name,e.name AS relation"
        #
        if relation == None:
            nGQL1 = f"MATCH (v:entity{{name: '{entity_name}'}})<-[e:relation]-(v2) RETURN v2.name AS s_Name, e.name AS relation, v.name AS d_Name"
            nGQL2 = f"MATCH (v:entity{{name: '{entity_name}'}})-[e:relation]->(v2) RETURN v.name AS s_Name, e.name AS relation, v2.name AS d_Name"
            nGQLs = [nGQL1, nGQL2]
        else:
            nGQL1 = f"MATCH (v:entity{{name: '{entity_name}'}})<-[e:relation{{name: '{relation}'}}]-(v2) RETURN v2.name AS s_Name, e.name AS relation, v.name AS d_Name"
            nGQL2 = f"MATCH (v:entity{{name: '{entity_name}'}})->[e:relation{{name: '{relation}'}}]-(v2) RETURN v.name AS s_Name, e.name AS relation, v2.name AS d_Name"
            nGQLs = [nGQL1, nGQL2]
        self.result = []
        for nGQL in nGQLs:
            # parse_Res(self.session.execute(nGQL))
            # print(self.session.execute(nGQL))
            self.result.extend(parse_Res(self.session.execute(nGQL)))

    def search(self, entity_name, relation=None, space_id=1):
        self.use_space(space_id=space_id)
        self.get_all_neigh(entity_name=entity_name, relation=relation)
        # result:s_name, d_name, relation
        # for s,d,r in self.result:
        #     print(f'src name: {s},relation: {r},dst name: {d}')
        # self.session.release()
        return self.result

    def insert(self, h_entity, relation, t_entity, space_id):
        self.use_space(space_id=space_id)
        h_vid = str(murmur64(bytes(h_entity, encoding="utf8"), seed=0xC70F6907))
        t_vid = str(murmur64(bytes(t_entity, encoding="utf8"), seed=0xC70F6907))
        search_nql = f"MATCH (v:entity{{name: '{h_entity}'}})-[e:relation]-(v2:entity{{name: '{t_entity}'}}) RETURN v.name AS s_Name, v2.name AS d_Name,e.name AS relation"
        rank = self.session.execute(search_nql).row_size()
        NQLs = []
        NQLs.append(f"UPSERT VERTEX ON entity {h_vid} SET name = '{h_entity}'")
        NQLs.append(f"UPSERT VERTEX ON entity {t_vid} SET name = '{t_entity}'")
        NQLs.append(
            f"UPSERT EDGE on relation {h_vid} -> {t_vid}@{rank} SET name = '{relation}'"
        )
        NQLs.append("REBUILD TAG INDEX value_index")
        NQLs.append("REBUILD TAG INDEX edge_index")
        for nql in NQLs:
            self.session.execute(nql)

    def delete(self, h_entity, relation, t_entity, space_id):
        self.use_space(space_id=space_id)
        h_vid = str(murmur64(bytes(h_entity, encoding="utf8"), seed=0xC70F6907))
        t_vid = str(murmur64(bytes(t_entity, encoding="utf8"), seed=0xC70F6907))
        NQLs = []
        search_nql = f"MATCH (v:entity{{name: '{h_entity}'}})-[e:relation]-(v2:entity{{name: '{t_entity}'}}) RETURN v.name AS s_Name, v2.name AS d_Name,e.name AS relation"
        rank = self.session.execute(search_nql).row_size()
        # print(search_nql, rank)
        for r in range(rank):
            NQLs.append(f"DELETE EDGE relation {h_vid} -> {t_vid}@{r}")
        for nql in NQLs:
            # print(nql)
            self.session.execute(nql)


if __name__ == "__main__":
    nebula_db = nebula_database()
    search_res = nebula_db.search(entity_name='灰铸铁', relation=None, space_id=1)
    print(search_res)
    nebula_db.session.release()
