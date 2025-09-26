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


# --------------------------- DP + Baseline Wrapper ----------------------------------------------

from pathlib import Path
from typing import List, Dict, Any, Tuple
import math

import spacy  # UD dependencies for DP

from .base import BaseSystem


# ----- DP helpers (Qiu et al., 2011; mapped to UD deps) -----

JJ = {"ADJ"}
NN = {"NOUN", "PROPN"}
PRON_TARGETS = {"it", "they"}

def load_lexicon(path_pos, path_neg):
    pos = {w.strip().lower() for w in Path(path_pos).read_text(encoding="utf-8", errors="ignore").splitlines()
           if w and not w.startswith(";")}
    neg = {w.strip().lower() for w in Path(path_neg).read_text(encoding="utf-8", errors="ignore").splitlines()
           if w and not w.startswith(";")}
    return pos, neg

def is_negated(tok):
    return any(c.dep_ == "neg" for c in tok.children) or tok.dep_ == "neg" or any(a.dep_ == "neg" for a in tok.ancestors)

def normalize_aspect(tok):
    lem = tok.lemma_.strip().lower()
    if lem in PRON_TARGETS:
        return None
    if tok.pos_ in NN and lem.isalpha():
        return lem
    return None

def adjective_form(tok):
    if tok.pos_ in JJ and tok.lemma_.isalpha():
        return tok.lemma_.lower()
    return None

def dp_extract_from_doc(doc, seeds_pos, seeds_neg, max_iters=5):
    """Return (targets, opinions{word->+1/-1}) using DP rules R11–R42 mapped to UD."""
    O: Dict[str, int] = {}
    T: set = set()

    # init opinions from seeds present in doc
    for t in doc:
        w = adjective_form(t)
        if not w: 
            continue
        if w in seeds_pos: O[w] = +1
        elif w in seeds_neg: O[w] = -1

    def adopt(word, ori):
        if word not in O:
            O[word] = ori
            return True
        return False

    changed = True
    it = 0
    while changed and it < max_iters:
        changed = False
        it += 1

        # R11/R12: opinions -> targets (amod; copular ADJ with nsubj)
        for tok in doc:
            w = adjective_form(tok)
            if not w or w not in O: 
                continue
            # amod: ADJ -> NOUN
            if tok.dep_ == "amod" and tok.head.pos_ in NN:
                a = normalize_aspect(tok.head)
                if a and a not in T:
                    T.add(a); changed = True
            # copular pred: ADJ head with nsubj NOUN
            if tok.pos_ == "ADJ":
                for child in tok.children:
                    if child.dep_ == "nsubj" and child.pos_ in NN:
                        a = normalize_aspect(child)
                        if a and a not in T:
                            T.add(a); changed = True

        # R31/R32: targets -> targets (conj siblings, same-head same-dep)
        for tok in doc:
            if tok.pos_ in NN and normalize_aspect(tok) in T:
                # conj siblings
                for sib in tok.conjuncts:
                    if sib.pos_ in NN:
                        a = normalize_aspect(sib)
                        if a and a not in T:
                            T.add(a); changed = True
                # same-head same-dep nouns
                if tok.head and tok.dep_ in {"nmod", "obl", "obj"}:
                    for sib in tok.head.children:
                        if sib is tok: 
                            continue
                        if sib.dep_ == tok.dep_ and sib.pos_ in NN:
                            a = normalize_aspect(sib)
                            if a and a not in T:
                                T.add(a); changed = True

        # R41/R42: opinions -> opinions (ADJ conj; co-modifiers of same NOUN)
        for tok in doc:
            if tok.pos_ != "ADJ":
                continue
            w = adjective_form(tok)
            if not w or w not in O: 
                continue
            # conj
            for sib in tok.conjuncts:
                ws = adjective_form(sib)
                if ws:
                    changed |= adopt(ws, O[w])
            # co-amod on same noun
            if tok.head and tok.dep_ == "amod":
                for sib in tok.head.children:
                    if sib is tok: 
                        continue
                    if sib.dep_ == "amod" and sib.pos_ == "ADJ":
                        ws = adjective_form(sib)
                        if ws:
                            changed |= adopt(ws, O[w])

        # R21/R22: targets -> opinions (amod on T; copular ADJ with T as nsubj)
        for tok in doc:
            if tok.pos_ in NN and normalize_aspect(tok) in T:
                # amod children
                for child in tok.children:
                    if child.dep_ == "amod" and child.pos_ == "ADJ":
                        w = adjective_form(child)
                        if w:
                            pol = +1
                            if is_negated(child): pol = -pol
                            changed |= adopt(w, pol)
                # copular predicate
                head = tok.head
                if head and head.pos_ == "ADJ" and any(c.dep_ == "nsubj" and c is tok for c in head.children):
                    w = adjective_form(head)
                    if w:
                        pol = +1
                        if is_negated(head): pol = -pol
                        changed |= adopt(w, pol)

    # drop pronouns as targets
    T = {t for t in T if t not in PRON_TARGETS}
    return T, O

def aggregate_review_aspect_labels(doc, targets, opinions,
                                   pos_seeds, neg_seeds, neutral_band=0.33):
    """Return {aspect -> {1,0,NaN}} via sentiment ratio over local opinion links."""
    counts = {a: {"pos": 0, "neg": 0} for a in targets}
    seedset = set(opinions.keys()) | pos_seeds | neg_seeds

    for sent in doc.sents:
        for tok in sent:
            a = normalize_aspect(tok)
            if not a or a not in targets:
                continue
            # amod child adjectives
            for child in tok.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    w = adjective_form(child)
                    if not w or w not in seedset:
                        continue
                    pol = opinions.get(w, (+1 if w in pos_seeds else -1))
                    if is_negated(child): pol = -pol
                    if pol > 0: counts[a]["pos"] += 1
                    else: counts[a]["neg"] += 1
            # copular predicate ADJ head with nsubj = tok
            head = tok.head
            if head and head.pos_ == "ADJ" and any(c.dep_ == "nsubj" and c is tok for c in head.children):
                w = adjective_form(head)
                if w and w in seedset:
                    pol = opinions.get(w, (+1 if w in pos_seeds else -1))
                    if is_negated(head): pol = -pol
                    if pol > 0: counts[a]["pos"] += 1
                    else: counts[a]["neg"] += 1

    labels: Dict[str, float] = {}
    for a, c in counts.items():
        total = c["pos"] + c["neg"]
        if total == 0:
            labels[a] = np.nan
        else:
            score = (c["pos"] - c["neg"]) / float(total)
            if score >= neutral_band: labels[a] = 1.0
            elif score <= -neutral_band: labels[a] = 0.0
            else: labels[a] = np.nan
    return labels


class SULMBaseline(BaseSystem):
    """
    Strict SULM baseline with Double Propagation (DP) preprocessing:
      • Runs DP on UD dependencies (spaCy) to extract aspects + propagate opinion seeds.
      • Aggregates per-review aspect polarities to {1,0,NaN}.
      • Trains SULM on rows: [user, item, overall(0/1), s_aspect1..K].
      • Saves/loads SULM shards + a meta.json (vocab, priors, hparams).
      • Serves predict_given_aspects → scores in [-1, 1] (one per requested aspect).
    """

    def __init__(self, args, reviews, tests):
        super().__init__(args, reviews, tests)

        # paths
        cache_dir = getattr(self.args, "cache_dir", "cache")
        div_name  = getattr(self.args, "div_name", "default")
        self.bundle_dir   = Path(cache_dir) / f"sulm_{div_name}"
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path    = self.bundle_dir / "meta.json"
        self.model_prefix = self.bundle_dir / "model_"

        # knobs
        self.max_vocab          = getattr(self.args, "sulm_max_vocab", 200)
        self.neutral_band       = getattr(self.args, "sulm_neutral_band", 0.33)
        self.overall_threshold  = getattr(self.args, "sulm_overall_threshold", 4.0)
        self.dp_max_iters       = getattr(self.args, "dp_max_iters", 5)
        self.dp_spacy_model     = getattr(self.args, "dp_spacy_model", "en_core_web_sm")
        self.dp_pos_lex         = getattr(self.args, "dp_pos_lex")
        self.dp_neg_lex         = getattr(self.args, "dp_neg_lex")

        self.num_factors = getattr(self.args, "sulm_num_factors", 8)
        self.iterations  = getattr(self.args, "sulm_iterations", 30)
        self.alpha       = getattr(self.args, "sulm_alpha", 0.5)
        self.lambda_b    = getattr(self.args, "sulm_lambda_b", 0.01)
        self.lambda_pq   = getattr(self.args, "sulm_lambda_pq", 0.01)
        self.lambda_z    = getattr(self.args, "sulm_lambda_z", 0.08)
        self.lambda_w    = getattr(self.args, "sulm_lambda_w", 0.01)
        self.gamma       = getattr(self.args, "sulm_gamma", 0.001)
        self.use_mult    = getattr(self.args, "sulm_mult", False)

        # logger
        self.logger = logging.getLogger("sulm")
        self.logger.setLevel(logging.INFO)

        # trained state
        self.model = None
        self.vocab: List[str] = []
        self.a2i: Dict[str, int] = {}
        self.aspect_priors: List[float] = []

        self._load_or_train()

    # ------------------------- API -------------------------

    def predict_given_aspects(self, user_id, item_id, aspects):
        r_prob, s_probs = self._predict_base(user_id, item_id)
        if len(s_probs) == len(self.vocab) + 1:
            s_probs = s_probs[:-1]
        assert len(s_probs) == len(self.vocab), "SULM returned unexpected sentiment vector length"
        scores = 2.0 * np.asarray(s_probs, dtype=float) - 1.0
        out: List[float] = []
        for name in aspects:
            j = self._lookup_index(name)
            if j is not None:
                out.append(float(scores[j]))
            else:
                # neutral fallback; swap to (2*r_prob-1) if you prefer overall influence
                out.append(0.0)
        return out

    # ------------------------- train/load -------------------------

    def _bundle_ready(self):
        if not self.meta_path.exists():
            return False
        shards = ["mu", "av_sent", "z", "user_profiles", "item_profiles"]
        return all(Path(str(self.model_prefix) + s).exists() for s in shards)

    def _load_or_train(self):
        if self._bundle_ready():
            meta = json.load(open(self.meta_path))
            self.vocab = list(meta["vocab"])
            self.aspect_priors = list(meta["priors"])
            self.a2i = {a: i for i, a in enumerate(self.vocab)}
            self.model = SentimentUtilityLogisticModel(
                logger=self.logger,
                ratings=[],
                num_aspects=len(self.vocab),
                num_factors=meta["num_factors"],
                lambda_b=meta["lambda_b"],
                lambda_pq=meta["lambda_pq"],
                lambda_z=meta["lambda_z"],
                lambda_w=meta["lambda_w"],
                gamma=meta["gamma"],
                iterations=meta["iterations"],
                alpha=meta["alpha"],
                l1=False, l2=True, mult=meta["mult"],
            )
            self.model.load(str(self.model_prefix))
            return

        # DP prep → ratings
        ratings, vocab, priors = self._build_ratings_with_dp(self.reviews)
        assert ratings and vocab, "No training data after DP; cannot train SULM."

        self.vocab = vocab
        self.aspect_priors = priors
        self.a2i = {a: i for i, a in enumerate(self.vocab)}

        # train
        np.random.seed(241)
        self.model = SentimentUtilityLogisticModel(
            logger=self.logger,
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
            l1=False, l2=True, mult=self.use_mult,
        )
        self.model.train_model(l1=False, l2=True)

        # save
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

    # ------------------------- DP → ratings -------------------------

    def _build_ratings_with_dp(self, reviews):
        assert self.dp_pos_lex and self.dp_neg_lex, "Set args.dp_pos_lex and args.dp_neg_lex (Hu–Liu lists)."
        pos_seeds, neg_seeds = load_lexicon(self.dp_pos_lex, self.dp_neg_lex)

        nlp = spacy.load(self.dp_spacy_model, disable=["ner"])
        nlp.enable_pipe("senter")

        aspect_df = {}
        aspect_counts: Dict[str, int] = {}
        rows_tmp: List[Tuple[dict, Dict[str, float], float]] = []

        for r in reviews:
            text = str(r.get("text") or "")
            if not text.strip():
                continue
            doc = nlp(text)

            # sentence-level DP then aggregate
            T_all, O_all = set(), {}
            for sent in doc.sents:
                T, O = dp_extract_from_doc(sent.as_doc(), pos_seeds, neg_seeds, max_iters=self.dp_max_iters)
                T_all |= T
                O_all.update(O)

            labels = aggregate_review_aspect_labels(doc, T_all, O_all, pos_seeds, neg_seeds, neutral_band=self.neutral_band)
            if not labels:
                continue

            # update counts
            for a in labels.keys():
                aspect_counts[a] = aspect_counts.get(a, 0) + 1

            overall = self._overall_binary(r, labels)
            rows_tmp.append((r, labels, overall))

        # fix vocab = top-K
        vocab = [a for a, _ in sorted(aspect_counts.items(), key=lambda kv: (-kv[1], kv[0]))][: self.max_vocab]
        a2i = {a: i for i, a in enumerate(vocab)}

        # assemble SULM rows + priors
        rows: List[List[Any]] = []
        col_bins: List[List[float]] = [[] for _ in vocab]

        for r, lab, overall in rows_tmp:
            uid, iid = r.get("user_id"), r.get("item_id")
            vals: List[float] = []
            for a in vocab:
                b = lab.get(a, np.nan)
                vals.append(b)
                if not (isinstance(b, float) and math.isnan(b)):
                    col_bins[a2i[a]].append(b)
            rows.append([uid, iid, overall] + vals)

        priors = [float(np.nanmean(np.array(col_bins[j], dtype=float))) if col_bins[j] else 0.5
                  for j in range(len(vocab))]
        return rows, vocab, priors

    # ------------------------- helpers -------------------------

    def _predict_base(self, user, item):
        assert self.model is not None, "Model not trained/loaded."
        r_prob, s_probs = self.model.predict(user, item)
        return float(r_prob), list(map(float, s_probs))

    def _lookup_index(self, name):
        a = (name or "").strip()
        if a in self.a2i:
            return self.a2i[a]
        a_low = a.lower()
        if a_low in self.a2i:
            return self.a2i[a_low]
        return None

    def _overall_binary(self, r, a2s):
        # 1) explicit overall_binary
        if "overall_binary" in r and r["overall_binary"] in (0,1):
            return float(r["overall_binary"])
        # 2) stars (0..5)
        for k in ("stars", "overall", "rating"):
            if k in r and r[k] is not None:
                try:
                    v = float(r[k])
                    if 0.0 <= v <= 5.0:
                        return 1.0 if v >= self.overall_threshold else 0.0
                    return 1.0 if v > 0.0 else 0.0
                except Exception:
                    pass
        # 3) fallback: mean sign of aspect labels
        vals = [(+1 if v == 1.0 else (-1 if v == 0.0 else 0)) for v in a2s.values()]
        return 1.0 if (np.mean(vals) if vals else 0.0) > 0 else 0.0
