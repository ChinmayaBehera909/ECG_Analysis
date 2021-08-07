import numpy as np
import json
import os

def secs_to_hms(seconds):

    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    x = str(int(hours))+":"+str(int(minutes))+":"+str(int(seconds))
    return x


def storeData(path, ecgObj, fileObj):

    try:
        path = os.join(path, fileObj.fileID)
        os.mkdir(path)
    except Exception as e:
        print('OS ERROR:', e)
        return

    #store cleaned signal
    np.save(path+fileObj.fileID+'.npy',ecgObj.data)


    #store peak informations
    hr = []
    for i in ecgObj.peaks:
        hr.append(ecgObj.rate[i])
    ecgObj.ecg_info['Heart_Rate'] = np.array(hr)
    if ecgObj.ecg_info:
        ecgObj.ecg_info['Peaks'] = ecgObj.peaks
    np.save(path+fileObj.fileID+'_peaks.npy',ecgObj.ecg_info)    # while loading the data can be accessed through .item() method


    #store ecg templates
    np.save(path+fileObj.fileID+'_templates.npy',ecgObj.ecg_templates)


    #store meta information
    meta={
        'File_ID' : fileObj.fileID,
        'File_Date': fileObj.fileDate.date().strftime("%m/%d/%Y"),
        'File_Time': fileObj.fileDate.time().strftime("%H:%M:%S"),
        'File_Type': fileObj.fileType,
        'File_Meta': fileObj.meta,
        'Recording_Type': ecgObj.sig_type,
        'Recording_Date': ecgObj.base_date.strftime("%m/%d/%Y"),
        'Recording_Time': ecgObj.base_time.strftime("%H:%M:%S"),
        'Recording_Duration': secs_to_hms(ecgObj.sig_len/ecgObj.fs),
        'Recording_FS': ecgObj.fs
    }
    json_object = json.dumps(meta, indent = 4)
    with open(path+fileObj.fileID+"_meta.json", "w") as outfile:
        outfile.write(json_object)

