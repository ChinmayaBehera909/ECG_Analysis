import numpy as np


class readFile:

    def __init__(self, path, fileDate, fileID, fileType):

        # instantiate the object with basic information
        self.path = path
        self.fileType = fileType
        self.fileDate = fileDate
        
        self.data = None
        
        self.fileID = fileID

        if self.fileType.lower() ==  'mit':

            self.data, self.meta = self.read_MIT(path)

        elif self.fileType.lower() ==  'edf':

            self.data, self.meta = self.read_EDF(path)

        else:
            print('Filetype not supported !!')

        if self.data == None:
            print('Failed to instatiate file object....ensure you enter correct parameters !!')

    def read_MIT(self, path):

        import wfdb

        try:
            record = wfdb.rdsamp(path)

            return np.asarray(record[0], dtype='float32'), record[1]

        except Exception as e:
            print('ERROR:',e)

            return None, None

    def read_EDF(self, path):

        import mne

        try:
            if path != None :
                
                edf = mne.io.read_raw_edf(path)

                data=edf.get_data().T
                
                meta = edf.info
                
                if len(meta['ch_names']) != 1:
                    print('Multiple channels detected:', meta['ch_names'])
                    ch_name = input('Select the channel number you want to proceed with:')
                    if ch_name.isdigit():
                        data=data[:,ch_name]
            
            return np.array(data, dtype='float32'),meta

        except Exception as e:
            print('ERROR: ',e)
            return None, None
        
        


