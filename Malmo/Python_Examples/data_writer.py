import os

class DataWriter:
    def __init__(self, data_type, data_path = 'default'):
        if not data_type in ['pathfinding_train_data']:
            raise Exception('Invalid data type for DataWriter: {}'.format(data_type))
        self.data_type = data_type
        self.data_path = 'data/{}/{}'.format(self.data_type, data_path)
        self.file_path = ''
        self.check_path()


    def check_path(self):
        ''' Establishes necessary directories
        '''
        if not os.path.exists('data/{}'.format(self.data_type)):
            os.mkdir('data/{}'.format(self.data_type))

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)


    def create_csv_record(self):
        ''' Creates pathfinding record csv file
        '''
        contents = ''
        if self.data_type == 'pathfinding_train_data':
            dir_contents = [d for d in os.listdir(self.data_path)]
            if len(dir_contents) == 0:
                self.file_path = self.data_path + '/pathfinding_0.csv' 
            else:
                for d in dir_contents:
                    if os.path.isfile(os.path.join(self.data_path, d)):
                        num = int(d.split('_')[-1].split('.')[0])
                        self.file_path = self.data_path + '/pathfinding_{}.csv'.format(num+1)
            contents = 'epNum,stepNum,posX,posZ,destX,destZ,action,reward,success,dist\n'
        
        f = open(self.file_path, 'w')
        f.write(contents)
        f.close()


    def record_csv_data(self, *args):
        ''' Writes args sequentially into line of csv
        '''
        contents = ''
        for arg in args:
            contents += '{},'.format(arg)
        contents = contents[:-1] + '\n'

        f = open(self.file_path, 'a')
        f.write(contents)
        f.close()



    
