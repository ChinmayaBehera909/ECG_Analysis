import numpy as np
import json

def milli_to_hms(millis):

    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))%24

    x = str(hours)+":"+str(minutes)+":"+str(seconds)
    return x


def storeData(path, ecgObj, fileObj):

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
        'Recording_Duration': milli_to_hms(ecgObj.fs*ecgObj.sig_len),
        'Recording_FS': ecgObj.fs
    }
    json_object = json.dumps(meta, indent = 4)
    with open(path+fileObj.fileID+"_meta.json", "w") as outfile:
        outfile.write(json_object)

