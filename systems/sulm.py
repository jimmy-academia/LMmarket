# https://github.com/kobauman/SULM/blob/master/sulm.py

import time
import sys
import numpy as np
import pandas as pd
import logging
import pickle
import json


'''
SentimentUtilityLogisticModel class implements
    Sentiment Utility Logistic Model (SULM).
This model estimates users' and items' profiles based on information extracted
from user reviews.
The data should contain:
    - userID
    - itemID
    - overallRating - {0,1}
    - list of aspects sentiments - {0,1,nan}
'''
class SentimentUtilityLogisticModel():
    '''
    ratings - the list of data points 
    num_aspects - number of aspects in the dataset
    num_factors - number of latent factors in the model
    lambda_b, lambda_pq - regularization parameter for profile coefficients (default=0.6)
    lambda_z, lambda_w  - regularization parameter for regression weight (default=0.6)
    gamma - the coefficient for the initial gradient descent step (default=1.0)
    iterations - number of iterations for training the model (default=30)
    alpha - the relative importance between rating and sentiment estimation parts (default=0.5)
    l1 - L1 normalization
    l2 - L2 normalization
    mult - multiplication of general-user-item coefficients
    '''
    def __init__(self, logger, ratings, num_aspects,
                 num_factors=3,
                 lambda_b=0.5,
                 lambda_pq=0.5,
                 lambda_z=0.5,
                 lambda_w=0.5,
                 lambda_su=0.05,
                 gamma=1.0,
                 iterations=30,
                 alpha=0.5,
                 l1=False,
                 l2=True,
                 mult=False):

        self.logger = logger
        self.ratings = ratings
        self.num_ratings = len(ratings)
        self.num_aspects = num_aspects
        self.num_factors = num_factors
        self.iterations = iterations
        self.alpha = alpha
        self.lambda_b = lambda_b
        self.lambda_pq = lambda_pq
        self.lambda_z = lambda_z
        self.lambda_w = lambda_w
        self.lambda_su = lambda_su
        self.gamma = gamma
        self.mu = None
        self.l1 = l1
        self.l2 = l2
        self.mult = mult
        self.average_sentiments()

    '''
        Create new profile
        user:   True - user profile; False - item profile
        random: True - random initial coefficients; False: profile coefficiets set to zeros
    '''
    def new_profile(self, profile_id, user=True, random=True):
        if user:
            self.profile_users[profile_id] = dict()
            if random:
                self.profile_users[profile_id]['bu'] = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1))
                self.profile_users[profile_id]['p']  = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1,self.num_factors))
                self.profile_users[profile_id]['w']  = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1))
            else:
                self.profile_users[profile_id]['bu'] = np.zeros(shape=(self.num_aspects+1))
                self.profile_users[profile_id]['p']  = np.zeros(shape=(self.num_aspects+1,self.num_factors))
                self.profile_users[profile_id]['w']  = np.zeros(shape=(self.num_aspects+1))
        else:
            self.profile_items[profile_id] = dict()
            if random:
                self.profile_items[profile_id]['bi'] = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1))
                self.profile_items[profile_id]['q']  = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1,self.num_factors))
                self.profile_items[profile_id]['v']  = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1))
            else:
                self.profile_items[profile_id]['bi'] = np.zeros(shape=(self.num_aspects+1))
                self.profile_items[profile_id]['q']  = np.zeros(shape=(self.num_aspects+1,self.num_factors))
                self.profile_items[profile_id]['v']  = np.zeros(shape=(self.num_aspects+1))
    
    def new_variable_profile(self, profile_id, user=True, random=True):
        variable_profile = dict()
        if user:
            variable_profile['bu'] = np.zeros(shape=(self.num_aspects+1))
            variable_profile['p']  = np.zeros(shape=(self.num_aspects+1,self.num_factors))
            variable_profile['w']  = np.zeros(shape=(self.num_aspects+1))
        else:
            variable_profile['bi'] = np.zeros(shape=(self.num_aspects+1))
            variable_profile['q']  = np.zeros(shape=(self.num_aspects+1,self.num_factors))
            variable_profile['v']  = np.zeros(shape=(self.num_aspects+1))
        return variable_profile.copy()
    
    
    
    '''Calculate average mu for parameter initialization'''
    def mu_initialization(self):
        if self.mu:
            return
        
        'iterate over aspects'
        aspects = zip(*self.ratings)
        self.mu = list()
        for i, aspect in enumerate(aspects):
            'rating'
            if i ==2:
                mean_rating = self.logistic_inverse(np.nanmean(aspect))
            'dismiss user_id, item_id, rating'
            if i < 3:
                continue
            self.mu.append(self.logistic_inverse(np.nanmean(aspect)))
        'last mu is for constant'
        self.mu.append(mean_rating)
        self.mu = np.array(self.mu)

    '''Calculate average sentiments'''
    def average_sentiments(self):
        aspects = zip(*self.ratings)
        self.avg_sentiments = list()
        for i, aspect in enumerate(aspects):
            'dismiss user_id, item_id, rating'
            if i < 3:
                continue
            self.avg_sentiments.append(np.nanmean(aspect))

    '''Calculate correlation between sentiments'''
    def sentiments_correlation(self):
        df = pd.DataFrame(self.ratings)
        corr = df.corr()
        df_len = len(df)
        for column in df.columns:
            frequency = len(df[df[column].notnull()])
            print('aspect: %d,\tfrequency: %d,\tpercent: %.2f' % (column, frequency, frequency/df_len*100))
            if column in corr:
                aspect_corr = corr[(corr[column] > 0.5) | (corr[column] < -0.5)][column]
                for aspect2 in aspect_corr.index:
                    if aspect2 != column:
                        aspect2_corr = aspect_corr.ix[aspect2,column]
                        collective_frequency = len(df[(df[column].notnull())&(df[aspect2].notnull())])
                        aspect2_frequency = len(df[df[aspect2].notnull()])
                        print('(%d,%d)\t%.3f\t%.2f\t(%.2f)'%(column,aspect2,
                                                             aspect2_corr,
                                                             100*collective_frequency/df_len,
                                                             100*aspect2_frequency/df_len))
                        
                            
    '''Train the model to fit the rating data set'''
    def train_model(self, l1 = False, l2 = True):
        #initialize coefficients
        self.mu_initialization() #initialize with average values
#         self.mu = np.random.normal(size=(self.num_aspects+1)) #random initialization
        self.logger.info('Initial mu: %s'%str(self.mu))
        self.z  = np.random.normal(loc=(1.0/self.num_aspects), scale=0.1, size=(self.num_aspects+1)) #random initialization
        self.profile_users = dict()
        self.profile_items = dict()
        Q_old = 100000000000000000000.0
        conv_num = 0
        #make the specified number of iterations
        for i in range(self.iterations):
            t0 = time.time()
            #self.ratings - the list of arrays
            #shuffle the list of ratings on each iteration
            np.random.shuffle(self.ratings)
            for num, element in enumerate(self.ratings):
                user = element[0]
                item = element[1]
                if user not in self.profile_users:
                    self.new_profile(user, user=True)
                if item not in self.profile_items:
                    self.new_profile(item, user=False)
                    
                rating = element[2]
                aspect_ratings = np.append(element[3:],np.nan)
                assert len(aspect_ratings) == self.num_aspects + 1

                # identify which aspects are specified
                indicator = np.invert(np.isnan(aspect_ratings))

                #calculate aspect sentiment predictions
                sentiment_utility_prediction  = self.calculate_sentiment_utility_prediction(user, item)
                # sentiment_utility_prediction_initial = sentiment_utility_prediction
                sentiment_prediction  = self.logistic(sentiment_utility_prediction)
                #calculate rating predictions
                if self.mult:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
                else:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
                rating_prediction = self.logistic(rating_utility_prediction)
                
                # calculate deltas
                delta_s = aspect_ratings - sentiment_prediction
                delta_s = delta_s - (np.abs(delta_s) < 0.001)*delta_s
                delta_s = delta_s - 0.001*(np.abs(delta_s) > 0.999)*delta_s
                
                delta_r = rating - rating_prediction
                delta_r = delta_r - (np.abs(delta_r) < 0.001)*delta_r
                delta_r = delta_r - 0.001*(np.abs(delta_r) > 0.999)*delta_r
                
                # update vector mu
                if self.mult:
                    mu_step = self.alpha * delta_r * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v'])
                else:
                    mu_step = self.alpha * delta_r * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v'])
                mu_step += (1 - self.alpha) * np.nan_to_num(indicator.astype(int) * delta_s)
                if any(np.abs(mu_step) > 1000):
                    print(mu_step)
                    print('mu_step',delta_r, self.z, self.profile_users[user]['w'], self.profile_items[item]['v'])
                    print()
                    exit()
                
                # Fix items and update users' profiles
                # take step towards negative gradient of log likelihood
                # we take a step in negative direction because we are minimizing functional Q
                bu_step = mu_step
                if self.l2:
                    self.profile_users[user]['bu'] -= self.gamma * self.lambda_b * self.profile_users[user]['bu']
                if self.l1:
                    self.profile_users[user]['bu'] -= self.gamma * self.lambda_b * np.sign(self.profile_users[user]['bu'])
                if self.lambda_su:
                    self.profile_users[user]['bu'] -= self.gamma * self.lambda_su * indicator.astype(int) * self.profile_users[user]['bu']
                
                self.profile_users[user]['bu'] += self.gamma * bu_step
                 
                p_step = np.matrix([np.dot(self.profile_items[item]['q'][i], mu_step[i]) for i in range(self.num_aspects+1)])
                if self.l2:
                    self.profile_users[user]['p'] -= self.gamma * self.lambda_pq * self.profile_users[user]['p']
                if self.l1:
                    self.profile_users[user]['p'] -= self.gamma * np.sign(self.profile_users[user]['p'])
                if self.lambda_su:
                    self.profile_users[user]['p'] -= self.gamma * self.lambda_su * np.matrix([np.dot(self.profile_items[item]['q'][i], indicator.astype(int)[i]) for i in range(self.num_aspects+1)]) 
                    
                
                self.profile_users[user]['p'] += self.gamma * p_step


                # Fix users and update items' profiles
                # take step towards negative gradient of log likelihood
                # we take a step in negative direction because we are minimizing functional Q
                #calculate aspect sentiment predictions
                sentiment_utility_prediction = self.calculate_sentiment_utility_prediction(user, item)
                sentiment_prediction = self.logistic(sentiment_utility_prediction)
                if self.mult:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
                else:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
                rating_prediction = self.logistic(rating_utility_prediction)
                 
                delta_s = aspect_ratings - sentiment_prediction
                delta_s = delta_s - (np.abs(delta_s) < 0.001)*delta_s
                delta_s = delta_s - 0.001*(np.abs(delta_s) > 0.999)*delta_s

                delta_r = rating - rating_prediction
                delta_r = delta_r - (np.abs(delta_r) < 0.001)*delta_r
                delta_r = delta_r - 0.001*(np.abs(delta_r) > 0.999)*delta_r
                
                # calculate mu_step
                if self.mult:
                    mu_step = self.alpha * delta_r * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v'])
                else:
                    mu_step = self.alpha * delta_r * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v'])
                mu_step += (1 - self.alpha) * np.nan_to_num(indicator.astype(int) * delta_s)
                
                bi_step = mu_step
                if self.l2:
                    self.profile_items[item]['bi'] -= self.gamma * self.lambda_b * self.profile_items[item]['bi']
                if self.l1:
                    self.profile_items[item]['bi'] -= self.gamma * self.lambda_b * np.sign(self.profile_items[item]['bi'])
                if self.lambda_su:
                    self.profile_items[item]['bi'] -= self.gamma * self.lambda_su * indicator.astype(int) * self.profile_items[item]['bi']
                
                self.profile_items[item]['bi'] += self.gamma * bi_step
                 
                q_step = np.matrix([np.dot(self.profile_users[user]['p'][i], mu_step[i]) for i in range(self.num_aspects+1)])
                if self.l2:
                    self.profile_items[item]['q'] -= self.gamma * self.lambda_pq * self.profile_items[item]['q']
                if self.l1:
                    self.profile_items[item]['q'] -= self.gamma * self.lambda_pq * np.sign(self.profile_items[item]['q'])
                if self.lambda_su:
                    self.profile_items[item]['q'] -= self.gamma * self.lambda_su * np.matrix([np.dot(self.profile_users[user]['p'][i], indicator.astype(int)[i]) for i in range(self.num_aspects+1)])
                    
                self.profile_items[item]['q'] += self.gamma * q_step
                
            
                # Fix users, items profiles and solve for weights
                # take step towards negative gradient of log likelihood
                # we take a step in negative direction because we are minimizing functional Q
                #calculate aspect sentiment predictions
                sentiment_utility_prediction  = self.calculate_sentiment_utility_prediction(user, item)
                sentiment_prediction  = self.logistic(sentiment_utility_prediction)
                #calculate rating predictions
                if self.mult:
                    rating_utility_prediction = sum(sentiment_utility_prediction*(self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
                else:
                    rating_utility_prediction = sum(sentiment_utility_prediction*(self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
                rating_prediction = self.logistic(rating_utility_prediction)
                
                
                delta_r = rating - rating_prediction
                delta_r = delta_r - (np.abs(delta_r) < 0.001)*delta_r
                delta_r = delta_r - 0.001*(np.abs(delta_r) > 0.999)*delta_r
                
                z_step = self.alpha * sentiment_utility_prediction * delta_r
                if self.mult:
                    z_step *= self.profile_users[user]['w']*self.profile_items[item]['v']
                
                if self.l2:
                    z_step -= self.lambda_z * self.z
                if self.l1:
                    z_step -= self.lambda_z * np.sign(self.z)
                
                w_step = self.alpha * sentiment_utility_prediction * delta_r
                if self.mult:
                    w_step *=  self.z*self.profile_items[item]['v']
                
                if self.l2:
                    w_step -= self.lambda_w * self.profile_users[user]['w']
                if self.l1:
                    w_step -= self.lambda_w * np.sign(self.profile_users[user]['w'])
                
                v_step = self.alpha * sentiment_utility_prediction * delta_r
                if self.mult:
                    v_step *= self.z*self.profile_users[user]['w']
                    
                if self.l2:
                    v_step -= self.lambda_w * self.profile_items[item]['v']
                if self.l1:
                    v_step -= self.lambda_w * np.sign(self.profile_items[item]['v'])
                
                
                self.z += self.gamma * z_step
                
                self.profile_users[user]['w'] += self.gamma * w_step
                
                self.profile_items[item]['v'] += self.gamma * v_step
                                 
                
                
                if num%10000==0 and num > 0:
                    self.logger.debug('%d elements processed'%num)

            #update the length of gradient descent step
            self.gamma *= 0.91
            t1 = time.time()
            Q_new = self.calculate_Q()
            Q_dif = (Q_old-Q_new)/Q_old
            Q_old = Q_new
            self.logger.info('Iteration %.2i finished in %.2f seconds with Q = %.3f (diff = %.4f)'% (i + 1, t1 - t0, Q_old, Q_dif))
            if Q_dif < 0.005 and Q_dif > 0:
                conv_num += 1
                if conv_num > 2:
                    self.logger.info('Model converged on iteration %.2i'%(i+1))
                    break
            else:
                conv_num = 0
    
    
    
    '''Calculate the aspect sentiments predictions based on user and item profile'''   
    def calculate_sentiment_utility_prediction(self, user, item):
        sentiment_utility_predictions  = self.mu.copy()
        sentiment_utility_predictions += self.profile_users[user]['bu']
        sentiment_utility_predictions += self.profile_items[item]['bi']
        product = [np.dot(self.profile_users[user]['p'][i],self.profile_items[item]['q'][i]) for i in range(self.num_aspects+1)]
        sentiment_utility_predictions += product
        return sentiment_utility_predictions
    
    
    '''Calculate the logistic function and its inverse'''
    def logistic(self,t):
        return 1/(1+np.exp(-t))
    def logistic_inverse(self,t):
        if t<0.00001:
            return -40
        return -np.log(1/t - 1)
    
    
    '''Calculate the value of the functional Q to be optimized'''
    def calculate_Q(self):
        rating_part = - self.log_likelihood_rating()
        sentiment_part = - self.log_likelihood_sentiment()
        rerularization_part = self.regularization()

        Q =  self.alpha * rating_part + (1 - self.alpha) * sentiment_part + rerularization_part
        if np.isnan(Q):
            print(rating_part, self.alpha, sentiment_part, rerularization_part)
        return Q
    
    
    '''Calculate log-likelihood for the sentiment part of the model'''
    def log_likelihood_sentiment(self):
        log_likelihood = 0
        for element in self.ratings:
            user = element[0]
            item = element[1]
            aspect_ratings = element[3:]
            
            indicator = np.invert(np.isnan(aspect_ratings))
            'calculate aspect sentiment predictions'
            s_logistic_predictions  = self.logistic(self.calculate_sentiment_utility_prediction(user, item))
#             print(s_logistic_predictions,aspect_ratings)
            for i in range(len(indicator)):
                if indicator[i]:
                    log_likelihood += aspect_ratings[i] * np.log(s_logistic_predictions[i])
                    log_likelihood += (1 - aspect_ratings[i]) * np.log(1 - s_logistic_predictions[i])
#             print('log_likelihood_sentiment',log_likelihood)
        return log_likelihood
    
    
    '''Calculate log-likelihood for the rating part of the model'''
    def log_likelihood_rating(self):
        log_likelihood = 0
        for element in self.ratings:
            user = element[0]
            item = element[1]
            rating = element[2]
            'calculate aspect sentiment predictions'
            s_predictions  = self.calculate_sentiment_utility_prediction(user, item)
            if self.mult:
                r_prediction = sum(s_predictions * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
            else:
                r_prediction = sum(s_predictions * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
            if np.isnan(r_prediction):
                print(s_predictions,self.z,self.profile_users[user]['w'],self.profile_items[item]['v'])
                exit()
            r_logistic_prediction = self.logistic(r_prediction)
            if r_logistic_prediction > 0.999:
                r_logistic_prediction = 0.999
            elif r_logistic_prediction < 0.001:
                r_logistic_prediction = 0.001
            log_likelihood += rating * np.log(r_logistic_prediction)
            log_likelihood += (1 - rating) * np.log(1-r_logistic_prediction)
            if np.isnan(log_likelihood):
                print(r_prediction,r_logistic_prediction)
                print(rating,np.log(r_logistic_prediction),np.log(1-r_logistic_prediction))
                exit()
                break
        return log_likelihood
    
    ''' Calculate the L2 regularization part of the model'''
    def regularization(self):
        user_norm = dict()
        item_norm = dict()
        
        if self.l2:
            norm_function = np.square
        elif self.l1:
            norm_function = np.abs
        
        norm_z = np.sum(norm_function(self.z))
        
        for user in self.profile_users:
            norm_b  = np.sum(norm_function(self.profile_users[user]['bu']))
            norm_pq = np.sum(norm_function(self.profile_users[user]['p']))
            norm_w  = np.sum(norm_function(self.profile_users[user]['w']))
            user_norm[user] = self.lambda_b * norm_b + self.lambda_pq * norm_pq +  self.lambda_w * norm_w
            
        for item in self.profile_items:
            norm_b  = np.sum(norm_function(self.profile_items[item]['bi']))
            norm_pq = np.sum(norm_function(self.profile_items[item]['q']))
            norm_w  = np.sum(norm_function(self.profile_items[item]['v']))
            item_norm[item] = self.lambda_b * norm_b + self.lambda_pq * norm_pq +  self.lambda_w * norm_w
        
        total_norm = 0
        for element in self.ratings:
            user = element[0]
            item = element[1]
            total_norm += user_norm[user] + item_norm[item] + self.lambda_z * norm_z
            
            aspect_ratings = np.append(element[3:],np.nan)
            indicator = np.invert(np.isnan(aspect_ratings))
            sentiment_utility_predictions  = indicator.astype(int) * self.calculate_sentiment_utility_prediction(user, item)
            total_norm += self.lambda_su * np.sum(np.square(sentiment_utility_predictions))
        return total_norm
    
    # Print train output ONLY for testing purposes 
    def predict_train(self):
        for i, element in enumerate(self.ratings):
            user = element[0]
            item = element[1]
#             rating = element[2]
            #calculate aspect sentiment predictions
            s_predictions  = self.calculate_sentiment_utility_prediction(user, item)
            s_logistic_predictions  = self.logistic(s_predictions)
            if self.mult:
                r_prediction = sum(s_predictions*(self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
            else:
                r_prediction = sum(s_predictions*(self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
            r_logistic_prediction = self.logistic(r_prediction)
            message = str(element) + '\nRating: %d\tPrediction: %.3f'%(element[2],r_logistic_prediction)
            message += '\nReal sentiments: %s\nPredicted sentiments: %s'%(element[3:],s_logistic_predictions[:-1])
            self.logger.info(message)
            if i > 15:
                break
            
    
    
    def predict(self, user, item):
        '''
        Predict ratings and sentiments for a pair of user and item
        Input:  user_id, item_id
        Output: rating_prediction, list of sentiment_predictions
        '''
        if user not in self.profile_users:
            self.new_profile(user, user=True, random=False)
        if item not in self.profile_items:
            self.new_profile(item, user=False, random=False)
            
        'calculate aspect sentiment predictions'
        sentiment_utility_prediction = self.calculate_sentiment_utility_prediction(user, item)
        sentiment_predictions  = self.logistic(sentiment_utility_prediction)
        if self.mult:
            rating_utility_prediction = sum(sentiment_utility_prediction * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
        else:
            rating_utility_prediction = sum(sentiment_utility_prediction * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
        rating_prediction = self.logistic(rating_utility_prediction)
        return rating_prediction, sentiment_predictions
    
    
    def calculate_aspect_impacts(self, user, item, average = False, absolute = True):
        '''Calculate aspect impacts for a given pair of user_id, item_id'''
        if user not in self.profile_users:
            self.new_profile(user, user = True, random = False)
        if item not in self.profile_items:
            self.new_profile(item, user = False, random = False)
            
        'calculate aspect sentiment predictions'
        sentiment_prediction  = self.logistic(self.calculate_sentiment_utility_prediction(user, item)[:-1])
        
        if self.mult:
            aspect_impacts = (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v'])[:-1]    
        else:
            aspect_impacts = (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v'])[:-1]

        if average:
            sentiment_difference = sentiment_prediction - self.avg_sentiments
            aspect_impacts = aspect_impacts * sentiment_difference
        else:
            aspect_impacts = aspect_impacts * sentiment_prediction
        
        if absolute:
            aspect_impacts =  np.abs(aspect_impacts)
        return list(aspect_impacts)
        
        
    
    '''Predict ratings and sentiments for a given dataset'''
    def predict_test(self, testset, filename):
        result = list()
        for element in testset:
            user = element[0]
            item = element[1]
            if user not in self.profile_users:
                self.new_profile(user, user=True, random=False)
            if item not in self.profile_items:
                self.new_profile(item, user=False, random=False)
            
            r_logistic_prediction, s_logistic_predictions = self.predict(user, item)
            result.append([user,item,r_logistic_prediction]+s_logistic_predictions.tolist())
            
            self.logger.info('User: %s, Item %s'%(user,item))
            self.logger.info('Aspect_impacts: '+
                             str(self.calculate_aspect_impacts(user, item, average = True, absolute = False)))
            self.logger.info('ABSOLUTE_Aspect_impacts: '+
                             str(self.calculate_aspect_impacts(user, item, average = True, absolute = True)))
            
        json.dump(result,open(filename,'w'))
        return result

    '''Print the model to file in the readable format'''
    def pretty_save(self,filename):
        model_file = open(filename,'w')
        model_file.write('mu = '+np.array_str(self.mu)+'\n')
        for user in self.profile_users:
            model_file.write('\n***********\n')
            model_file.write(user+'\nbu = '+np.array_str(self.profile_users[user]['bu'])+'\n')
            model_file.write('p = '+np.array_str(self.profile_users[user]['p'])+'\n')
            model_file.write('w = '+np.array_str(self.profile_users[user]['w'])+'\n')
        model_file.write('\n===========================\n===========================\n\n')
        for item in self.profile_items:
            model_file.write('\n***********\n')
            model_file.write(item+'\nbi = '+np.array_str(self.profile_items[item]['bi'])+'\n')
            model_file.write('q = '+np.array_str(self.profile_items[item]['q'])+'\n')
            model_file.write('v = '+np.array_str(self.profile_items[item]['v'])+'\n')
        model_file.write('\n===========================\n===========================\n\n')
        model_file.write('z = '+np.array_str(self.z)+'\n')
        model_file.close()
        
    '''Save the model'''
    def save(self, filename):
        pickle.dump(self.mu, open(filename+'mu', 'wb'))
        pickle.dump(self.avg_sentiments, open(filename+'av_sent', 'wb'))
        pickle.dump(self.z, open(filename+'z', 'wb'))
        pickle.dump(self.profile_users, open(filename+'user_profiles', 'wb'))
        pickle.dump(self.profile_items, open(filename+'item_profiles', 'wb'))
    
    '''Load the model'''
    def load(self, filename):
        self.mu = pickle.load(open(filename+'mu', 'rb'))
        self.avg_sentiments = pickle.load(open(filename+'av_sent', 'rb'))
        self.z = pickle.load(open(filename+'z', 'rb'))
        self.profile_users = pickle.load(open(filename+'user_profiles', 'rb'))
        self.profile_items = pickle.load(open(filename+'item_profiles', 'rb'))



     
# if __name__ == '__main__':
#     logger = logging.getLogger('signature')
#     logging.basicConfig(format='%(asctime)s : %(name)-12s: %(levelname)s : %(message)s')
#     logging.root.setLevel(level=logging.DEBUG)
#     logger.info("running %s" % ' '.join(sys.argv))
    
#     ratings = [['user1','item1',1,1,1,0],
#                ['user1','item2',1,0,1,0],
#                ['user2','item1',0,1,0,1],
#                ['user2','item2',0,np.nan,0,1],
#                ['user2','item3',1,1,0,1],
#                ['user3','item1',1,np.nan,0,0],
#                ['user3','item2',0,np.nan,0,1],
#                ['user3','item3',1,1,1,1],
#                ['user4','item3',1,0,1,1],
#                ['user4','item1',0,1,0,1],
#                ['user4','item2',1,0,1,1]
#                ]
    
#     np.random.seed(241)
#     aspects = ['food','service','decor']
#     model = SentimentUtilityLogisticModel(logger, ratings, num_aspects=len(aspects), num_factors=5,
#                                           lambda_b = 0.05, lambda_pq = 0.05, lambda_z = 0.05, lambda_w = 0.05,
#                                           gamma=2.0, iterations=5, alpha=0.5, 
#                                           l1 = False, l2 = True, mult = False)
    
# #     model.sentiments_correlation()
#     model.train_model()

#     logger.info('Average Sentiments:\n%s'%str(list(zip(aspects, model.avg_sentiments))))
#     model.pretty_save('readable_model.txt')
#     model.predict_train()
#     model.save('model_test_')

#     modelnew = SentimentUtilityLogisticModel(logger, ratings,num_aspects=len(aspects), num_factors=5,
#                                              lambda_b = 0.01, lambda_pq = 0.01, lambda_z = 0.08, lambda_w = 0.01,
#                                              gamma=0.001,iterations=30, alpha=0.00)
#     modelnew.load('model_test_')

#     testset = [['user1','item3'],
#                ['user1','item4'],
#                ['user2','item4']
#                ]
#     modelnew.predict_test(testset,'model_test.txt')

from pathlib import Path
from typing import List, Dict, Any, Tuple
import math
import logging

import torch
from sentence_transformers import SentenceTransformer

from .base import BaseSystem
from .ou import OUBaseline


class SULMBaseline(BaseSystem):
    """
    Retrofit of kobauman/SULM to your BaseSystem:
      • Builds a fixed aspect vocab from OU-labeled reviews (top-K by frequency).
      • Trains SULM on rows: [user_id, item_id, overall(0/1), s_aspect1..K] with s ∈ {1,0,nan}.
      • Predicts per requested aspects; unknown aspect names are mapped to nearest vocab entry
        via MiniLM embeddings (cosine similarity threshold).
      • Returns scores in [-1, 1], matching your evaluator.

    Disk layout (under cache_dir / sulm_{div_name}):
      meta.json
      model_{{mu,av_sent,z,user_profiles,item_profiles}}   # SULM's own save shards
    """

    # ------------------------- public API -------------------------

    def __init__(self, args, reviews, tests):
        super().__init__(args, reviews, tests)

        # paths
        cache_dir = getattr(self.args, "cache_dir", "cache")
        div_name  = getattr(self.args, "div_name", "default")
        self.bundle_dir   = Path(cache_dir) / f"sulm_{div_name}"
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path    = self.bundle_dir / "meta.json"
        self.model_prefix = self.bundle_dir / "model_"

        # knobs (sane baselines; override via CLI/args)
        self.max_vocab     = getattr(self.args, "sulm_max_vocab", 200)
        self.label_limit   = getattr(self.args, "sulm_label_limit", 4000)
        self.neutral_band  = getattr(self.args, "sulm_neutral_band", 0.33)
        self.map_threshold = getattr(self.args, "sulm_map_threshold", 0.45)

        self.num_factors = getattr(self.args, "sulm_num_factors", 8)
        self.iterations  = getattr(self.args, "sulm_iterations", 30)
        self.alpha       = getattr(self.args, "sulm_alpha", 0.5)
        self.lambda_b    = getattr(self.args, "sulm_lambda_b", 0.01)
        self.lambda_pq   = getattr(self.args, "sulm_lambda_pq", 0.01)
        self.lambda_z    = getattr(self.args, "sulm_lambda_z", 0.08)
        self.lambda_w    = getattr(self.args, "sulm_lambda_w", 0.01)
        self.gamma       = getattr(self.args, "sulm_gamma", 0.001)
        self.use_mult    = getattr(self.args, "sulm_mult", False)

        # encoder for aspect-name fallback mapping
        model_name = getattr(self.args, "aspect_encoder_name", "sentence-transformers/all-MiniLM-L6-v2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(model_name, device=device)

        # will be set by load/train
        self.model = None
        self.vocab: List[str] = []
        self.a2i: Dict[str, int] = {}
        self.vocab_embeds = None
        self.aspect_priors: List[float] = []

        self._load_or_train()

    def predict_given_aspects(self, user_id: str, item_id: str, aspects: List[str]) -> List[float]:
        """
        Return one score per input aspect in [-1, 1].
        """
        r_prob, s_probs = self._predict_base(user_id, item_id)
        # drop the constant slot if present
        if len(s_probs) == len(self.vocab) + 1:
            s_probs = s_probs[:-1]
        assert len(s_probs) == len(self.vocab), "SULM returned unexpected sentiment vector length"
        s_scores = 2.0 * s_probs - 1.0  # map [0,1] -> [-1,1]

        out: List[float] = []
        for name in aspects:
            j = self._aspect_to_index(name)
            if j is not None:
                out.append(float(s_scores[j]))
            else:
                # conservative fallback to overall: [-1,1] centered at 0
                out.append(float(2.0 * r_prob - 1.0))
        return out

    # ------------------------- training / loading -------------------------

    def _load_or_train(self) -> None:
        if self._bundle_ready():
            meta = json.load(open(self.meta_path))
            self.vocab = list(meta["vocab"])
            self.aspect_priors = list(meta["priors"])
            self.a2i = {a: i for i, a in enumerate(self.vocab)}

            logger = logging.getLogger("sulm")
            logger.setLevel(logging.INFO)
            # Create a lightweight shell and load trained weights
            self.model = SentimentUtilityLogisticModel(
                logger=logger,
                ratings=[],  # placeholder
                num_aspects=len(self.vocab),
                num_factors=meta["num_factors"],
                lambda_b=meta["lambda_b"],
                lambda_pq=meta["lambda_pq"],
                lambda_z=meta["lambda_z"],
                lambda_w=meta["lambda_w"],
                gamma=meta["gamma"],
                iterations=meta["iterations"],
                alpha=meta["alpha"],
                l1=False,
                l2=True,
                mult=meta["mult"],
            )
            self.model.load(str(self.model_prefix))
            self._encode_vocab()
            return

        # Ensure labels exist (annotate a subset with OU if missing)
        self._ensure_labels()

        # Build ratings + vocab from labeled reviews
        ratings, vocab, priors = self._build_ratings(self.reviews)
        assert len(ratings) > 0 and len(vocab) > 0, "No training data after labeling; cannot train SULM."
        self.vocab = vocab
        self.aspect_priors = priors
        self.a2i = {a: i for i, a in enumerate(self.vocab)}

        # Train SULM
        logger = logging.getLogger("sulm")
        logger.setLevel(logging.INFO)
        np.random.seed(241)
        self.model = SentimentUtilityLogisticModel(
            logger=logger,
            ratings=ratings,
            num_aspects=len(self.vocab),
            num_factors=self.num_factors,
            lambda_b=self.lambda_b,
            lambda_pq=self.lambda_pq,
            lambda_z=self.lambda_z,
            lambda_w=self.lambda_w,
            gamma=self.gamma,
            iterations=self.iterations,
            alpha=self.alpha,
            l1=False,
            l2=True,
            mult=self.use_mult,
        )
        self.model.train_model(l1=False, l2=True)

        # Save bundle
        meta = {
            "vocab": self.vocab,
            "priors": self.aspect_priors,
            "num_factors": self.num_factors,
            "lambda_b": self.lambda_b,
            "lambda_pq": self.lambda_pq,
            "lambda_z": self.lambda_z,
            "lambda_w": self.lambda_w,
            "gamma": self.gamma,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "mult": self.use_mult,
        }
        json.dump(meta, open(self.meta_path, "w"))
        self.model.save(str(self.model_prefix))
        self._encode_vocab()

    def _bundle_ready(self) -> bool:
        if not self.meta_path.exists():
            return False
        shards = ["mu", "av_sent", "z", "user_profiles", "item_profiles"]
        return all((self.model_prefix.as_posix() + s).startswith(self.model_prefix.as_posix()) or True
                   for s in shards) and all(Path(str(self.model_prefix) + s).exists() for s in shards)

    # ------------------------- label / ratings prep -------------------------

    def _ensure_labels(self) -> None:
        """
        If many reviews lack opinion_units, annotate up to label_limit via OU (in-place).
        """
        unlabeled = [r for r in self.reviews if not r.get("opinion_units")]
        if not unlabeled:
            return
        # deterministically pick a capped subset to label
        to_label = unlabeled[: self.label_limit]
        ou = OUBaseline(self.args, self.reviews, self.tests)
        ou.segmentation(to_label)  # expected to inject r["opinion_units"] entries

    def _build_ratings(self, reviews: List[dict]) -> Tuple[List[List[Any]], List[str], List[float]]:
        """
        Build SULM ratings rows + vocab + aspect priors.
        Row format: [user_id, item_id, overall_binary, s1..sK]
        """
        aspect_counts: Dict[str, int] = {}
        per_review: List[Tuple[dict, Dict[str, float], float]] = []

        for r in reviews:
            units = r.get("opinion_units") or []
            if not units:
                continue

            agg: Dict[str, List[float]] = {}
            for u in units:
                a = (u.get("aspect") or "").strip().lower()
                s = self._score_from_unit(u)
                if a:
                    agg.setdefault(a, []).append(s)
            if not agg:
                continue

            a2s = {a: float(np.mean(v)) for a, v in agg.items()}  # [-1,1]
            for a in a2s:
                aspect_counts[a] = aspect_counts.get(a, 0) + 1

            mean_sc = float(np.mean(list(a2s.values())))
            overall = 1.0 if mean_sc > 0.0 else 0.0

            per_review.append((r, a2s, overall))

        # vocab: top-K by frequency
        vocab = [a for a, _ in sorted(aspect_counts.items(), key=lambda kv: (-kv[1], kv[0]))][: self.max_vocab]
        a2i = {a: i for i, a in enumerate(vocab)}

        # rows + priors
        rows: List[List[Any]] = []
        col_bins: List[List[float]] = [[] for _ in vocab]

        for r, a2s, overall in per_review:
            row = [r.get("user_id"), r.get("item_id"), overall]
            vals: List[float] = []
            for a in vocab:
                if a in a2s:
                    v = a2s[a]
                    if v >= self.neutral_band:
                        b = 1.0
                    elif v <= -self.neutral_band:
                        b = 0.0
                    else:
                        b = np.nan
                    vals.append(b)
                    if not math.isnan(b):
                        col_bins[a2i[a]].append(b)
                else:
                    vals.append(np.nan)
            rows.append(row + vals)

        priors = [float(np.nanmean(np.array(col_bins[j], dtype=float))) if col_bins[j] else 0.5
                  for j in range(len(vocab))]

        return rows, vocab, priors

    @staticmethod
    def _score_from_unit(u: dict) -> float:
        """
        Map opinion unit to a numeric score in [-1,1].
        Prefer 'sentiment_score' if present; else coarse mapping from 'sentiment'.
        """
        if "sentiment_score" in u and u["sentiment_score"] is not None:
            return float(u["sentiment_score"])
        s = (u.get("sentiment") or "").strip().lower()
        if s.startswith("pos"):
            return 1.0
        if s.startswith("neg"):
            return -1.0
        return 0.0

    # ------------------------- prediction helpers -------------------------

    def _predict_base(self, user: str, item: str) -> Tuple[float, np.ndarray]:
        """
        Call SULM.predict; return rating prob (0..1) and per-aspect probs (np.array).
        """
        assert self.model is not None, "Model not trained/loaded."
        r_pred, s_preds = self.model.predict(user, item)
        return float(r_pred), np.asarray(s_preds, dtype=float)

    def _encode_vocab(self) -> None:
        """
        Precompute embeddings for vocab (unit-norm).
        """
        self.vocab_embeds = self.encoder.encode(self.vocab, convert_to_tensor=True, normalize_embeddings=True)

    def _aspect_to_index(self, name: str) -> int | None:
        """
        Exact/LC match → nearest neighbor via MiniLM (cosine) → None.
        """
        a = (name or "").strip()
        if a in self.a2i:
            return self.a2i[a]
        a_low = a.lower()
        if a_low in self.a2i:
            return self.a2i[a_low]
        if not self.vocab:
            return None

        q = self.encoder.encode([a_low], convert_to_tensor=True, normalize_embeddings=True)
        sims = (q @ self.vocab_embeds.T).squeeze(0)  # cosine because normalized
        j = int(torch.argmax(sims).item())
        if float(sims[j]) >= self.map_threshold:
            return j
        return None
