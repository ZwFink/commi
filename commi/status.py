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
