from nebula3.gclient.net import ConnectionPool, Session
from nebula3.storage import GraphStorageClient


class NebulaConnection:
    """Singleton 连接管理。"""
    _pool: ConnectionPool | None = None
    _storage: GraphStorageClient | None = None

    @classmethod
    def init(cls, hosts, user, password, space, **kwargs):
        from nebula3.Config import Config
        cfg = Config()
        cfg.max_connection_pool_size = kwargs.get("pool_size", 10)
        cls._pool = ConnectionPool()
        assert cls._pool.init(hosts, cfg), "Pool init failed"
        cls._space = space
        cls._user = user
        cls._password = password
        cls._storage = GraphStorageClient(hosts)

    @classmethod
    def session(cls) -> Session:
        sess = cls._pool.get_session(cls._user, cls._password)
        sess.execute(f"USE {cls._space}")
        return sess

    @classmethod
    def storage(cls) -> GraphStorageClient:
        return cls._storage
