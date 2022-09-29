from tools import Dao


class Infraction:
    def __init__(self):
        self.infrationID = None
        self.infrationTypes = ""
        self.time = None
        self.saveDIR = None
    # 违规类初始化


class Model:
    def __init__(self):
        self.modelID = None
        self.modelName = ""
        self.modelDIR = ""
        self.modelFunction = ""

    def setData(self, modelID, modelName, modelDIR, modelFunction):
        self.modelID = modelID
        self.modelName = modelName
        self.modelDIR = modelDIR
        self.modelFunction = modelFunction

    def insert(self):
        url = f'insert into model(`modelName`,`modelDIR`,`modelFunction`) value' \
              f'(\'{self.modelName}\',\'{self.modelDIR}\',\'{self.modelFunction}\')' \
              f'on duplicate key update `modelFunction` = \'{self.modelFunction}\''
        Dao.insert(url)


class Camera:
    def __init__(self):
        self.cameraID = None
        self.cameraUrl = ""
        self.name = ""
        self.order = None
        self.comment = ""

    def set(self, cameraID, cameraUrl, name, order, comment=""):
        self.cameraID = cameraID
        self.cameraUrl = cameraUrl
        self.name = name
        self.order = order
        self.comment = comment

    def insert(self):
        url = f'insert into camera(`cameraUrl`,`name`,`order`,`comment`) value' \
              f'(\'{self.cameraUrl}\',\'{self.name}\',{self.order},\'{self.comment}\')' \
              f'on duplicate key update `order` = \'{self.order}\',`comment` = \'{self.comment}\''
        Dao.insert(url)


class Fence:
    def __init__(self):
        self.fenceID = None
        self.fenceDIR = ""


class Detect:
    def __init__(self):
        self.cameraID = None
        self.modelID = None
        self.status = None
        self.comment = ""
        self.fenceID = None

    def setData(self, cameraID, modelID, status, comment, fenceID):
        self.cameraID = cameraID
        self.modelID = modelID
        self.status = status
        self.comment = comment
        self.fenceID = fenceID

    def insert(self):
        detectUrl = f'insert into detect(`cameraID`,`modelID`,`status`,`comment`) value' \
                    f'({self.cameraID},{self.modelID},{self.status},\'{self.comment}\')' \
                    f'on duplicate key update `status` = {self.status},`comment` = \'{self.comment}\''
        Dao.insert(detectUrl)
