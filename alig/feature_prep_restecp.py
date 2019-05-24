from category_encoders import *
import pandas as pd 
import numpy as np
from scipy.special import comb

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.autograd import Variable

class FeatureProcessor:
    def __init__(self,date_features, categorical_features,continous_features,label_column,max_cardinality=4,verbose=0):
        date_features.sort()
        categorical_features.sort()
        continous_features.sort()
        
        self.date_feats  = date_features
        self.cat_feats   = categorical_features
        self.conti_feats = continous_features
        self.label_column = label_column
        self.verbose      = verbose
        self.max_cardinality = max_cardinality

        
    def categorical_processor(self,train_X,train_y,cat_feats,label_column,test_X=None):

        """
        input : feature df and label column separtly
        output : do ohe for low cardinality features and catboost for the remaining.

        """
        if self.verbose>0:
            print('Me is here in categorical_processor')
        #ohe low cardinality
        low_cardinality_feats = [col for col in cat_feats if train_X[col].nunique() <= self.max_cardinality] 

        if len(low_cardinality_feats)>0:
            print('doing ohe..............')
            if self.verbose>0:
                print(low_cardinality_feats)
            ohe_lb = OneHotEncoder(cols=low_cardinality_feats)
            train_X = ohe_lb.fit_transform(train_X)
            if test_X:
                test_X  = ohe_lb.transform(test_X)

            cat_feats = list(set(cat_feats)-set(low_cardinality_feats))
            if len(cat_feats)<1:
                if test_X:

                    return train_X,test_X
                else:
                    return train_X

        #catboost feats remaining
        print('cat boost cols')
        print(cat_feats)
        catboost_lb = CatBoostEncoder(cols=cat_feats)
        train_X = catboost_lb.fit_transform(train_X,train_y)
        if test_X:
            if self.verbose>0:
                print('both train and test datasets are passed')
            X_test = catboost_lb.transform(X_test)
            return train_X,test_X
        else:
            if self.verbose>0:
                print('only train dataset is passed')
            return train_X


    def date_parser(self,df,date_features):

        if self.verbose>0:
            print('Me is here in date_parser')
        initial_no_columns = df.shape[1]
        combination_tracker = []
        for col in date_features:
            df[col] = pd.to_datetime(df[col])
        
        for col in date_features:
            df['day_{}'.format(col)] = df[col].dt.day
            df['month_{}'.format(col)] = df[col].dt.month
            df['year_{}'.format(col)] = df[col].dt.month
            for second_col in set(date_features)-set([col]):
                if set([second_col,col]) in combination_tracker:
                    continue
                else:
                    combination_tracker.append(set([second_col,col]))

                df['diff_{a}_{b}'.format(a=col,b=second_col)] = (df[col] - df[second_col]).dt.days
        
        #sanity check
        no_date_cols = len(date_features)
        new_cols_added_by_diff = comb(no_date_cols,2)
        day_month_type_features = 3*no_date_cols
        total_new_cols = new_cols_added_by_diff + day_month_type_features

        if df.shape[1]-initial_no_columns != total_new_cols:
            exit('somthing went wrong in date parsing')
        else:
            if self.verbose>0:
                print('date parsing completed sucessfully')
        if self.verbose>0:  
            print('no of new columns added for {a} date cols {b}'.format(a=len(date_features),b=total_new_cols))
        return df
    

    def continuous_categorical_maskup(self,train_X,cat_feats,conti_feats,test_X=None):
        """
        """
        if self.verbose>0:
            print('Me is here in cocontinuous_categorical_maskup')

        initial_columns = train_X.columns.tolist()
        for i,col in enumerate(cat_feats):
            if self.verbose>0:
                print('checking out {a}th ,ie {b} out of {c}'.format(a=i,b=col,c=len(cat_feats)))
            temp = train_X.groupby(by=col)[conti_feats].median(skipna=True).reset_index()
            temp.columns = [col+'_median_'+i if i!=col else col for i in temp.columns]
            train_X = pd.merge(train_X,temp,on=col,how='left')
            if test_X:
                test_X  = pd.merge(test_X,temp,on=col,how='left')

            temp = train_X.groupby(by=col)[conti_feats].max(skipna=True)
            temp.columns = [col+'_max_'+i if i!=col else col for i in temp.columns]
            train_X = pd.merge(train_X,temp,on=col,how='left')
            if test_X:
                test_X  = pd.merge(test_X,temp,on=col,how='left')

            temp = train_X.groupby(by=col)[conti_feats].min(skipna=True)
            temp.columns = [col+'_max_'+i if i!=col else col for i in temp.columns]
            train_X = pd.merge(train_X,temp,on=col,how='left')
            if test_X:
                test_X  = pd.merge(test_X,temp,on=col,how='left')

        #sanity check
        count_cols_added = 3*len(cat_feats)*len(conti_feats)
        new_cols_added = list(set(train_X.columns.tolist())-set(initial_columns))
        if new_cols_added !=count_cols_added:
            exit('Booyakasha! Check Dis! Column mismatch ')
        if self.verbose > 1:
            print('{a} columns added in continuous_categorical_maskup. They are {b}'.format(a=len(new_cols_added),b=new_cols_added))
        
        return train_X,test_X
    

    def arithmetic_mashup(self,train_X,train_y,conti_feats,test_X=None):
        """
        """
        for col in train_X.columns:
            med_val = train_X[col].median(skpna=True)
            train_X[col] = train_X[col].fillna(med_val)

            if test_X:
                test_X[col] = test_X[col].fillna(med_val)

        

        


    def deep_feat_generator(self,train_X,train_y,conti_feats,test_X=None):

    
        class Net(nn.Module):
            
            def __init__(self,continous_features):
                super(Net, self).__init__()
                n1 = len(continous_features)
                print(n1)
                self.fc1 = nn.Linear(n1,800)
                self.fc2 = nn.Linear(800,800)

                
            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.dropout(x, p=0.1)
                x = F.relu(x)
                x = self.fc2(x)
                
                return x


        x_train,y_train,x_test = train_X,train_y,test_X  

        #preprocessing for neural net


        for col in x_train.columns:
            med_val = train_X[col].median(skpna=True)
            x_train[col] = x_train[col].fillna(med_val)
            x_train[col] = x_train[col].fillna(med_val)

            train_min = x_train[col].min()
            train_max = x_train[col].max()
            x_train[col] -= train_min
            x_train[col] /= train_max

            if x_test:
                x_test[col] = x_test[col].fillna(med_val)
                x_test[col] -= train_min
                x_test[col] /= train_max
        
        #network training
        net = Net(continous_features=conti_feats)

        batch_size = 50
        num_epochs = 5
        learning_rate = 0.001
        batch_no = len(x_train) // batch_size

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        

        for epoch in range(num_epochs):

            # x_train, y_train = shuffle(x_train, y_train)
            # Mini batch learning
            for i in range(batch_no):
                start = i * batch_size
                end = start + batch_size

                x_var = Variable(torch.FloatTensor(x_train[start:end].values))
                y_var = Variable(torch.LongTensor(y_train[start:end].values))
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                ypred_var = net(x_var)
                loss =criterion(ypred_var, y_var)
                loss.backward()
                optimizer.step()
            # if epoch % 100 == 0:
            print('Epoch {}'.format(epoch+1))
            print(loss.detach().numpy())
    
        #getting features
        train_var = Variable(torch.FloatTensor(x_train.values), requires_grad=True)
        with torch.no_grad():
            result_train = net(train_var)
            
        out_size = 800
        train = pd.DataFrame(data=result_train.numpy(),columns=['deep_feat_{}'.format(i) for i in range(1,out_size+1)])
        # train['target'] = y_train
        print(train.head(2))

        if test_X:
            test_var = Variable(torch.FloatTensor(x_test.values), requires_grad=True)
            with torch.no_grad():
                result_test = net(test_var)

            test = pd.DataFrame(data=result_test.numpy(),columns=['deep_feat_{}'.format(i) for i in range(1,out_size+1)])
            # test['target'] = y_val
            print(test.head(2))
            exit('Not implemented')
        else:
            return train




    
    def fit_transform(self,train_X,train_y,test_X=None):
        # #all arithmetic operations
        # if test_X:
        #     pass
        # else:
        #     pass
        # #deep feat
        # if test_X:
        #     pass

        # else:
        #     print('only train')
        #     train_X_conti = self.deep_feat_generator(train_X=train_X[self.conti_feats],train_y=train_y,conti_feats=self.conti_feats,test_X=None)
        #     train_X = pd.concat([train_X,train_X_conti],axis=1)


        if test_X:
            train_X_cat, test_X_cat = self.categorical_processor(train_X=train_X[self.cat_feats],train_y=train_y,cat_feats=self.cat_feats,label_column=self.label_column,test_X=test_X[self.cat_feats])
            train_X = train_X.drop(self.cat_feats,axis=1)
            train_X = pd.concat([train_X,train_X_cat],axis=1)

            test_X = test_X.drop(self.cat_feats,axis=1)
            test_X = pd.concat([test_X,test_X_cat],axis=1)
        else:
            train_X_cat = self.categorical_processor(train_X=train_X[self.cat_feats],train_y=train_y,cat_feats=self.cat_feats,label_column=self.label_column,test_X=None)
            train_X = train_X.drop(self.cat_feats,axis=1)
            train_X = pd.concat([train_X,train_X_cat],axis=1)
        # print(train_X[self.cat_feats].head())
        # if test_X:
        #     train_X, test_X = self.continuous_categorical_maskup(train_X,self.cat_feats,self.conti_feats,test_X=test_X)
        # else:
        #     train_X = self.continuous_categorical_maskup(train_X,self.cat_feats,self.conti_feats)

            
        if test_X:
            df = pd.concat([train_X,test_X])
        else:
            train_X[self.label_column[0]]=train_y
        
        df = self.date_parser(df=train_X,date_features=self.date_feats)
    
        return df

