import scipy, math, scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import curve_fit

NUM_CHANNELS = 100
NUM_SAMPLES = 1000
t1 = 2*(2**.5)
t2 = 3*(2**.5)

def logistic_model(x, L, k, m):
	return L / (1 + np.exp(-1*k *(x-m)))

def s_curves(x, c=NUM_CHANNELS):


	f, ax = plt.subplots()
	thresholds = np.linspace(0, np.amax(x), num=NUM_SAMPLES)
	total_hits = []
	for i in range(c):
		b_x = x[i].copy()
		hits = []
		for j in range(NUM_SAMPLES):
			t = thresholds[j]
			h = np.count_nonzero(b_x > t)
			hits.append(h/NUM_SAMPLES)

			
			# b_x[b_x>=t] = 1
			# b_x[b_x!=1] = 0

		total_hits.append(np.asarray(hits))
		
		# ax.scatter(thresholds, hits, s=2)
		# ax.set_xlabel('Threshold')
		# ax.set_ylabel('Fraction of hits')
		# ax.set_title('.5 sigma uncorrelated')


		param_vals, param_covariances = curve_fit(logistic_model, thresholds, hits)
		y = logistic_model(thresholds, *param_vals)
		# ax.plot(thresholds, y, color='red',linewidth=2)
		
		# pedestal
		# ax.axvline(x=param_vals[2], color='green')
		# ax.axhline(y=0.5, color='green')
		
		# print('')
		# print(param_vals)
		# print(param_covariances)

	return np.asarray(total_hits)

def channel_mult():

	# correlated
	channels_hit1_test = []
	channels_hit2_test = []
	# uncorrelated
	channels_hit3 = []
	channels_hit4 = []
	# both
	channels_hit5 = []
	channels_hit6 = []

	# .5 correlated, 1.322 unc
	channels_hit7 = []
	channels_hit8 = []
	
	u_c = np.random.standard_normal(NUM_SAMPLES**2)
	data = np.random.standard_normal((NUM_CHANNELS,NUM_SAMPLES**2)) * (2**.5) 		# uncorrelated
	data1 = np.random.standard_normal(NUM_SAMPLES**2) * (2**.5)
	data2 = np.random.standard_normal((NUM_CHANNELS, NUM_SAMPLES**2)) 				# uncorrelated & correlated
	data3 = np.random.standard_normal((NUM_CHANNELS,NUM_SAMPLES**2)) * (0.) 		# correlated
	data4 = np.random.standard_normal((NUM_CHANNELS,NUM_SAMPLES**2)) * (1.75 ** .5) # .5 sigma unc

	for i in range(NUM_SAMPLES**2):
		data2[:,i] += u_c[i]
		data3[:,i] += data1[i]
		data4[:,i] += u_c[i] * .5
		# channels_hit1.append(np.count_nonzero(data1[i]>t1))
		# channels_hit2.append(np.count_nonzero(data1[i]>t2))
		channels_hit1_test.append(np.count_nonzero(data3[:,i]>t1))
		channels_hit2_test.append(np.count_nonzero(data3[:,i]>t2))

		channels_hit3.append(np.count_nonzero(data[:,i]>t1))
		channels_hit4.append(np.count_nonzero(data[:,i]>t2))

		channels_hit5.append(np.count_nonzero(data2[:,i]>t1))
		channels_hit6.append(np.count_nonzero(data2[:,i]>t2))

		channels_hit7.append(np.count_nonzero(data4[:,i]>t1))
		channels_hit8.append(np.count_nonzero(data4[:,i]>t2))
	return data, data3, data2, data4 
	f1, (ax1, ax2) = plt.subplots(1,2)

	b = np.arange(NUM_CHANNELS+1)
	ax1.hist(channels_hit5, label='.5 sigma uncorrelated', bins=b)
	ax1.hist(channels_hit7, label='half correlated', bins=b)
	ax1.hist(channels_hit1_test, label='only correlated', bins=b)
	ax1.hist(channels_hit3, label='only uncorrelated', bins=b)

	ax1.legend()
	ax1.set_yscale('log')
	ax1.set_title('2 sigma threshold')

	ax2.hist(channels_hit6, label='correlated & uncorrelated', bins=b)
	ax2.hist(channels_hit8, label='half correlated', bins=b)
	ax2.hist(channels_hit2_test, label='only correlated', bins=b)
	ax2.hist(channels_hit4, label='only uncorrelated', bins=b)

	ax2.legend()
	ax2.set_yscale('log')
	ax2.set_title('3 sigma threshold')
	

def pairwise(x, t, sigma):
	f, ax = plt.subplots()
	img = np.zeros((NUM_CHANNELS, NUM_CHANNELS))
	
	for j in range(0, NUM_CHANNELS-1):
		p_1 = np.count_nonzero(x[j] > t)
		hit_ind = np.argwhere(x[j] > t)
		# print(hit_ind.shape)
		for k in range(j+1, NUM_CHANNELS):
			n = 0
			for i in range(len(hit_ind)):
				ind = hit_ind[i]
				if x[k, ind] > t: n+=1
			# a = n * integrate_f(sigma, t)
			# img[j,k] = a * n
			img[j,k] = n 

	# np.fill_diagonal(img, 0)
	# img = img/np.mean(np.triu(img))
	img = img/np.mean(img)
	ax.imshow(img, cmap='gray')
	return img

# gaussian prob from t --> inf
def p(t, sigma):
	if sigma==2: return 0
	return integrate.quad(lambda x: np.exp((x**2) / (-2 * sigma**2))/((2*np.pi)**.5 * sigma),t,np.inf)[0]

# P(s_corr>t)*P(s>t-s_corr)
def f(s_corr, sigma_corr, t):
	return p(t, sigma_corr) * p(t-s_corr, sigma_corr)

# integrate above function from -inf --> inf
def integrate_f(sigma_corr, t):
	return integrate.quad(lambda s_corr: f(s_corr, sigma_corr, t),-np.inf,np.inf)[0]

# print(integrate_f(1, 2**.5))


# INTEGRAL VS SIGMA
def int_sig():

	sig_corr = np.linspace(0, 2**.5, 100)
	# sig_unc  = [(2-(i**2))**.5 for i in sig_corr][:-1] #(2 - (sig_corr**2))**.5
	# sig_unc.append(0.1)
	
	y  = [integrate_f(i, 2*(2**.5)) for i in sig_corr]
	y1 = [integrate_f(i, 3*(2**.5)) for i in sig_corr]
	y2 = [integrate_f(i, 2**.5) for i in sig_corr]
	y3 = [integrate_f(i, 1.5*(2**.5)) for i in sig_corr]

	# fi, (ax, ax1) = plt.subplots(1,2)
	fu, ax = plt.subplots()
	ax.scatter(sig_corr, y,  label='2*2**.5', s=5)
	ax.scatter(sig_corr, y1, label='3*2**.5', s=5)
	ax.scatter(sig_corr, y2, label='2**.5', s=5)
	ax.scatter(sig_corr, y3, label='1.5*2**.5', s=5)
	ax.set_title('P vs sigma_corr')
	ax.set_xlabel('sigma_corr')
	ax.set_ylabel('P')
	ax.legend()

	# y  = [integrate_f(i, 2*2**.5) for i in sig_unc]
	# y1 = [integrate_f(i, 3*2**.5) for i in sig_unc]
	# y2 = [integrate_f(i, 2**.5) for i in sig_unc]
	# y3 = [integrate_f(i, 1.5*2**.5) for i in sig_unc]


	# ax1.scatter(sig_unc, y,  label='2*2**.5', s=2)
	# ax1.scatter(sig_unc, y1, label='3*2**.5', s=2)
	# ax1.scatter(sig_unc, y2, label='2**.5', s=2)
	# ax1.scatter(sig_unc, y3, label='1.5*2**.5', s=2)
	# ax1.set_xlabel('sigma_unc')
	# ax1.set_ylabel('integral')
	# ax1.legend()


# INTEGRAL VS THERSHOLDS
def int_thres():
	thres = np.linspace(0, 5, 100)
	y  = [integrate_f(0.1, i) for i in thres]
	y1 = [integrate_f(.5, i) for i in thres]
	y2 = [integrate_f(.75, i) for i in thres]
	y3 = [integrate_f(1, i) for i in thres]
	y4 = [integrate_f(2**.5, i) for i in thres]


	f1, ax1 = plt.subplots()
	ax1.scatter(thres, y,  label='sigma = 0', s=2)
	ax1.scatter(thres, y1, label='sigma = .5', s=2)
	ax1.scatter(thres, y2, label='sigma = .75', s=2)
	ax1.scatter(thres, y3, label='sigma = 1', s=2)
	ax1.scatter(thres, y4, label='sigma = 2**.5', s=2)

	ax1.set_xlabel('thresholds')
	ax1.set_ylabel('integral')
	ax1.legend()


x_n = np.random.standard_normal((NUM_CHANNELS,NUM_SAMPLES)) * (2 ** .5) #+ np.random.multivariate_normal(mu, cov, size=NUM_SAMPLES).transpose()
# for i in range(NUM_SAMPLES):
# 	x_n[:,i] += u_c[i] * .5
# 	data3[:,i] += data1[i]
# 	data4[:,i] += u_c[i] * .5

x_n = channel_mult()

#pairwise(x_n[0], 3*(2**.5)) #sigma_corr = 0
# b = pairwise(x_n[2], 3*(2**.5)) #sigma_corr = 1

pairwise(x_n[3], 2*(2**.5), 1.75**.5) #sigma_corr = 1.32

int_sig()

plt.show()