import pandas as pd
import xml.etree.ElementTree as ET
import os
from lxml import etree


def parse_xml_file(file_path):
    # tree = ET.parse(file_path)
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    tree = ET.parse(file_path, parser=parser)
    root = tree.getroot()

    events = []
    timex3s = []
    tlinks = []
    sectimes = []
    summaries = []

    for event in root.findall(".//EVENT"):
        events.append({
            "filepath": file_path,
            "id": event.get("id"),
            "start": event.get("start"),
            "end": event.get("end"),
            "text": event.get("text"),
            "modality": event.get("modality"),
            "polarity": event.get("polarity"),
            "type": event.get("type"),
        })

    for timex3 in root.findall(".//TIMEX3"):
        timex3s.append({
            "filepath": file_path,
            "id": timex3.get("id"),
            "start": timex3.get("start"),
            "end": timex3.get("end"),
            "text": timex3.get("text"),
            "type": timex3.get("type"),
            "val": timex3.get("val"),
            "mod": timex3.get("mod"),
        })

    for tlink in root.findall(".//TLINK"):
        tlinks.append({
            "filepath": file_path,
            "id": tlink.get("id"),
            "fromID": tlink.get("fromID"),
            "fromText": tlink.get("fromText"),
            "toID": tlink.get("toID"),
            "toText": tlink.get("toText"),
            "type": tlink.get("type"),
        })

    for sectime in root.findall(".//SECTIME"):
        sectimes.append({
            "filepath": file_path,
            "id": sectime.get("id"),
            "start": sectime.get("start"),
            "end": sectime.get("end"),
            "text": sectime.get("text"),
            "type": sectime.get("type"),
            "dvalue": sectime.get("dvalue"),
        })

    for txt in root.findall("TEXT"):
        summaries.append({"filepath": file_path, "summary": txt.text})

    return summaries, events, timex3s, tlinks, sectimes


def xml_to_df(data_dir):
    # Parse all XML files and combine the data

    all_files = os.listdir(data_dir)
    xml_files = [
        data_dir + file for file in all_files if file.endswith(".xml")
    ]  # Test with all the files

    all_events = []
    all_timex3s = []
    all_tlinks = []
    all_sectimes = []
    all_summaries = []

    for file in xml_files:
        # print(file)
        summaries, events, timex3s, tlinks, sectime = parse_xml_file(file)
        all_summaries.extend(summaries)
        all_events.extend(events)
        all_timex3s.extend(timex3s)
        all_tlinks.extend(tlinks)
        all_sectimes.extend(sectime)

    # Create DataFrames
    summaries_df = pd.DataFrame(all_summaries)
    events_df = pd.DataFrame(all_events)
    timex3s_df = pd.DataFrame(all_timex3s)
    tlinks_df = pd.DataFrame(all_tlinks)
    sectimes_df = pd.DataFrame(all_sectimes)

    # sectime_indices = [index for index, value in tlinks_df['id'].items() if 'Sectime' in value]

    # tlinks_df = tlinks_df.drop(sectime_indices)

    return summaries_df, events_df, timex3s_df, tlinks_df, sectimes_df
