import psycopg2
import psycopg2.extensions


row_insert_query = "insert into uav{} (hbmx_count, hcmx_count, pbmx_count, pcmx_count, uav_id, flight_mode, " \
                   "hb1, hb2, hb3, hb4, " \
                   "hb1_fac, hb2_fac, hb3_fac, hb4_fac, " \
                   "hc1, hc2, hc3, hc4, " \
                   "hc1_fac, hc2_fac, hc3_fac, hc4_fac, " \
                   "psb, psb_fac, " \
                   "psc, psc_fac, " \
                   "bat_h, bat_lc," \
                   "hb1_fa, hb2_fa, hb3_fa, hb4_fa, " \
                   "hc1_fa, hc2_fa, hc3_fa, hc4_fa, " \
                   "psb_fa, psc_fa, " \
                   "no_missions, mission_mode, mission_id, mission_progress, rem_mission_len, health_index, " \
                   "failure_detection, critic_comp, rul,contingency, asset_risk, mission_risk) " \
                   "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, " \
                   "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, " \
                   "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, " \
                   "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, " \
                   "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"


def init_database(uav_count: int) -> psycopg2.extensions.connection:
    """Erstellt für jede Drohne eine Tabelle in der Datenbank

    Args:
        uav_count(int): Die Anzahl an Drohnen

    Returns:
        psycopg2.extensions.connection: Verbindung zur Datenbank

    """
    # connect to postgres
    con = psycopg2.connect(
        host="localhost",
        database="uavdatabase",
        user="postgres",
        password="kurabiyeci54"
    )
    # cursor
    cur = con.cursor()

    for u in range(uav_count):
        cur.execute(f"drop table if exists uav{u+1}")
        # create empty uav table
        cur.execute(
            f"create table if not exists uav{u+1} (index serial primary key, hbmx_count real, hcmx_count real, pbmx_count real, pcmx_count real, entry_time timestamp not null default current_timestamp, uav_id integer,"
            " flight_mode real, hb1 real, hb2 real, hb3 real, hb4 real, hb1_fac real, hb2_fac real, hb3_fac real, hb4_fac real, "
            "hc1 real, hc2 real, hc3 real, hc4 real, hc1_fac real, hc2_fac real, hc3_fac real, hc4_fac real, psb real, psb_fac real, psc real, psc_fac real, bat_h real, bat_lc real,"
            "hb1_fa real, hb2_fa real, hb3_fa real, hb4_fa real, hc1_fa real, hc2_fa real, hc3_fa real, hc4_fa real, psb_fa real, psc_fa real, no_missions integer, "
            "mission_mode real, mission_id real, mission_progress real, rem_mission_len real, health_index real, failure_detection text, critic_comp text, rul real,"
            "contingency text, asset_risk text, mission_risk text);")

    con.commit()
    cur.close()

    return con
