from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import time
import numpy as np
import pandas as pd
import holidays

tr_holidays = holidays.TR(list(range(2014, 2019)))
holiday_dates = []
for holiday_date, holiday_name in tr_holidays.items():
    holiday_dates.append(pd.Timestamp(holiday_date).date())


root_path = os.getcwd()

class EnergyDataset(Dataset):
    def __init__(self, 
                 args,
                 root_path, 
                 flag, 
                 size,
                 data_path,
                 target, 
                 scale, 
                 pre_transform_func, 
                 freq, 
                 seasonal_patterns=None):
        
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # init
        assert flag in ['train', 'val', 'test', 'future']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2, 'future':3}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        self.pre_transform_func = pre_transform_func
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        self.scaler = StandardScaler()

        if self.flag != 'future':
        
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col=0)

            if self.args.tag == 'alldata':
                border1s = [0]
                border2s = [len(df_raw)]
            else:
                border1s = [0, int(len(df_raw) * 0.8) - self.seq_len, int(len(df_raw) * 0.9)  - self.seq_len]
                border2s = [int(len(df_raw) * 0.8), int(len(df_raw) * 0.9), len(df_raw)]

            # border1s = [0, int(len(df_raw) * 0.9) - self.seq_len]
            # border2s = [int(len(df_raw) * 0.9), len(df_raw)]

            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        
        else:
            train_path = self.data_path.replace('test', 'train')

            df_train = pd.read_csv(os.path.join(self.root_path, train_path), index_col=0)
            df_future = pd.read_csv(os.path.join(self.root_path, self.data_path), index_col=0)
            df_raw = pd.concat((df_train, df_future)).reset_index()
            border1s = [0, len(df_train) - self.seq_len]
            border2s = [len(df_train), len(df_raw)]
 
            border1 = border1s[1]
            border2 = border2s[1]

        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_data, data_stamp = self.pre_transform_func(df_raw)
        
        cols = df_data.columns.tolist()
        cols.remove(self.target)
        cols.append(self.target)
        df_data = df_data[cols]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
 
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
def preprocess_wind_plant_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)     

    df['datetime'] = pd.to_datetime(df['datetime'])
    date_range = pd.date_range(df['datetime'].min(), df['datetime'].max(), freq='H')
    n_missing = len(date_range) - len(df)
    
    if n_missing != 0:
        print(f">>> {n_missing} records are missing, which will be filled with interpolation")
        df.set_index('datetime', inplace=True)
        new_df = pd.DataFrame(np.nan, index = date_range, columns = df.columns)
        new_df.index.name = 'datetime'
        new_df.loc[df.index] = df.values
        df = new_df.interpolate().reset_index()
    
    df['cos_windDir10m'] = np.cos(2 * np.pi * df['windDir10m'] / 360)
    df['cos_windDir100m'] = np.cos(2 * np.pi * df['windDir100m'] / 360)
 
    data_stamp = df['datetime'].to_frame()
    data_stamp['year'] = data_stamp['datetime'].dt.year
    data_stamp['cos_month'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
    data_stamp['cos_week'] = np.cos(2 * np.pi * df['datetime'].dt.isocalendar().week.astype(float) / 52)
    data_stamp['cos_day'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear / 365)
    data_stamp['cos_hour'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    data_stamp = data_stamp[['year', 'cos_month', 'cos_week', 'cos_day', 'cos_hour']].values
    
    df.set_index('datetime', inplace=True)
    
    return df, data_stamp

def preprocess_electricty_demand_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)     

    df['datetime'] = pd.to_datetime(df['datetime'])
    date_range = pd.date_range(df['datetime'].min(), df['datetime'].max(), freq='H')
    n_missing = len(date_range) - len(df)
    
    if n_missing != 0:
        print(f">>> {n_missing} records are missing, which will be filled with interpolation")
        df.set_index('datetime', inplace=True)
        new_df = pd.DataFrame(np.nan, index = date_range, columns = df.columns)
        new_df.index.name = 'datetime'
        new_df.loc[df.index] = df.values
        df = new_df.interpolate().reset_index()
    
    df['year'] = df['datetime'].dt.year.values
    df['cos_month'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
    df['cos_week'] = np.cos(2 * np.pi * df['datetime'].dt.isocalendar().week.astype(float) / 52)
    df['cos_day'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear / 365)
    df['cos_hour'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)

    df['is_holiday'] = df['datetime'].dt.date.isin(holiday_dates).astype(float)
    df['weekends_change_july_2017'] = ((df['datetime'] > pd.Timestamp(2017,7,1) ) & (df['datetime'].dt.day_name().isin(['Saturday', 'Sunday']))).astype(float)
    
    # weekday_onehotencoded = pd.get_dummies(df['datetime'].dt.weekday, prefix='weekday')
    # df = pd.concat((df, weekday_onehotencoded), axis=1)

    data_stamp = df[['year', 'cos_month', 'cos_week', 'cos_day', 'cos_hour']].copy()
    df.drop(data_stamp.columns, axis=1, inplace=True)
    df.set_index('datetime', inplace=True)
    
    return df, data_stamp.values



def data_provider(args, flag):
    
    pre_transform_func = preprocess_electricty_demand_data \
                        if args.data == 'electricity_demand' \
                        else preprocess_wind_plant_data 
                   
    dt = EnergyDataset(args,
                        root_path = root_path, #, 
                        flag = flag, 
                        size = [args.seq_len, args.label_len, args.pred_len], #[24, 1, 1],
                        data_path= f'data/{args.data}/train.csv' if flag != 'future' else f'data/{args.data}/test.csv',
                        target = args.target,#, 
                        scale = True, 
                        pre_transform_func = pre_transform_func, 
                        freq = args.freq)#'h')
    
    print(">>>",flag, len(dt))

    data_loader = DataLoader(
                            dt,
                            batch_size= args.batch_size if flag != 'future' else 1,
                            shuffle=True,
                            #pin_memory=False, num_workers=0,
                            #num_workers=args.num_workers,
                            drop_last = flag != 'future')
    
    args.input_size = dt.data_x.shape[1]
    
    return dt, data_loader
