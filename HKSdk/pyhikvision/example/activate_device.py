import os

from HKSdk.pyhikvision.hkws import base_adapter, config

# 初始化配置文件
cnf = config.Config()
path = os.path.join('../local_config.ini')
cnf.InitConfig(path)

# 初始化SDK适配器
adapter = base_adapter.BaseAdapter()
adapter.add_lib(cnf.SDKPath, cnf.Suffix)
adapter.init_sdk()
res = adapter.activate_device(cnf.IP, cnf.Port, cnf.Password)
print(res)