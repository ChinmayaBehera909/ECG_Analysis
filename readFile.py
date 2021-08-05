
class readFile:

    def __init__(self, path, fileDate, fileID, fileType):

        # instantiate the object with basic information
        self.path = path
        self.fileType = fileType
        self.fileDate = fileDate
        
        self.data = None
        
        self.fileID = fileID

        if self.fileType.lower() ==  'mit':

            self.data, self.meta = self.read_MIT()

        elif self.fileType.lower() ==  'edf':

            self.data, self.meta = self.read_EDF()

        else:
            raise TypeError('Filetype not supported !!')

        try:
            if self.data == None:
                print('Failed to instatiate file object....ensure you enter correct parameters !!')
        except:
            pass

    def read_MIT(self):

        import wfdb
        import numpy as np
        try:
            record = wfdb.rdsamp(self.path)
            data = np.asarray(record[0], dtype='float32')

            if len(record[1]['sig_name']) != 1:
                    print('Multiple channels detected:', record[1]['sig_name'])
                    ch_name = int(input('Select the channel number you want to proceed with:'))
                    data=data[:,ch_name]

            return data, record[1]

        except Exception as e:
            print('ERROR:',e)

            return None, None

    def read_EDF(self):

        import mne
        import numpy as np

        try:
            if self.path != None :
                
                edf = mne.io.read_raw_edf(self.path)

                data=edf.get_data().T
                
                meta = edf.info
                
                if len(meta['ch_names']) != 1:
                    print('Multiple channels detected:', meta['ch_names'])
                    ch_name = int(input('Select the channel number you want to proceed with:'))
                    data=data[:,ch_name]
            
            return np.array(data, dtype='float32'),meta

        except Exception as e:
            print('ERROR: ',e)
            return None, None
        
        


