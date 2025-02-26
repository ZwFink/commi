from dataclasses import dataclass

COMMI_STATUS_IGNORE = -1
COMMI_STATUSES_IGNORE = -1
@dataclass
class Status:
    count: int = 0
    cancelled: int = 0
    COMMI_SOURCE: int = 0
    COMMI_TAG: int = 0
    COMMI_ERROR: int = 0

    def get_source(self):
        return self.COMMI_SOURCE
    def get_tag(self):
        return self.COMMI_TAG

    @property
    def source(self):
        return self.COMMI_SOURCE

    @source.setter
    def source(self, v):
        self.COMMI_SOURCE = v

    @source.deleter
    def source(self):
        del self.COMMI_SOURCE

    @property
    def tag(self):
        return self.COMMI_TAG

    @tag.setter
    def tag(self, v):
        self.COMMI_TAG = v

    @tag.deleter
    def tag(self):
        del self.COMMI_TAG
    
