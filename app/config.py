import json

class Dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

class Config(object):
    @staticmethod
    def __load__(data):
        if type(data) is dict:
            return Config.load_dict(data)
        elif type(data) is list:
            return Config.load_list(data)
        else:
            return data

    @staticmethod
    def load_dict(data: dict):
        result = Dict()
        for key, value in data.items():
            result[key] = Config.__load__(value)
        return result

    @staticmethod
    def load_list(data: list):
        result = [Config.__load__(item) for item in data]
        return result

    @staticmethod
    def load_json(path: str):
        with open(path, "r") as f:
            result = Config.__load__(json.loads(f.read()))
        return result 
        



if __name__ == "__main__":
    conf = Config.load_json('config.json')
    print(conf.version)
    print(conf.bind.address)
    print(conf.bind.port)
    print(conf.security.area_working)
    print(conf.traffic.weights)
    print(conf.security.bootstrap_servers)

    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile) # Reading the file
        print("Read Config successful")
        print("data:",config)
        jsonfile.close()