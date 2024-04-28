from enum import Enum


DEL_MISSION_MIN_DISTANCE = 40  # Distanz Delivery-Mission: 40-60 km
DEL_MISSION_MAX_DISTANCE = 60
REC_MISSION_MIN_DISTANCE = 5  # Distanz Reconnessaince-Mission: 5-15 km
REC_MISSION_MAX_DISTANCE = 10



class MissionType(Enum):
    """Diese Enumerations-Klasse beschreibt die verschiedenen Arten von Missionen der Drohnen
        Delivery-Mission:
        Reconnaissance Mission:
    """
    DELIVERY_MISSION = 0
    RECONNAISSANCE_MISSION = 1


mission_duration = {MissionType.DELIVERY_MISSION: [DEL_MISSION_MIN_DISTANCE, DEL_MISSION_MAX_DISTANCE],
                    MissionType.RECONNAISSANCE_MISSION: [REC_MISSION_MIN_DISTANCE, REC_MISSION_MAX_DISTANCE]}
