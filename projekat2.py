import gym
import time
import numpy as np
import requests
import random
from PIL import Image
from gym.spaces import Box
from gym.spaces import Discrete
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
#pip install torch torchvision torchaudio
#pip install stable-baselines3[extra]
jar_path_igre = "D:\Java\JAVA_PROGRAMS\greed_island-master\greed_island-0.0.1-SNAPSHOT.jar"
#java -jar D:\Java\JAVA_PROGRAMS\greed_island-master\greed_island-0.0.1-SNAPSHOT.jar

class RafRpg(gym.Env):
  mapa_simbola = {
    '>': Image.open('Frontend/path/to/hill-image.png'),
    '<': Image.open('Frontend/path/to/revealed-hill-image.png'),
    '|': Image.open('Frontend/path/to/gate-image.png'),
    '$': Image.open('Frontend/path/to/mountain-image.png'),
    '_': Image.open('Frontend/path/to/meadow-image.png'),
    '.': Image.open('Frontend/path/to/revealed-meadow-image.png'),
    '-': Image.open('Frontend/path/to/water-image.png'),
    '+': Image.open('Frontend/path/to/forest-image.png'),
    ':': Image.open('Frontend/path/to/revealed-forest-image.png'),
    'V': Image.open('Frontend/path/to/farmer-image.png'),
    'B': Image.open('Frontend/path/to/bandit-image.png'),
    'M': Image.open('Frontend/path/to/trader-image.png'),
    'P': Image.open('Frontend/path/to/player-image.png'),
    'x': Image.open('Frontend/path/to/default-image.png')
    
    }
  def __init__(self) -> None:
    super().__init__()
    self.url_root = "http://localhost:8082"
    url = self.url_root+"/map/restart"
    payload={}
    headers = {}
    response = requests.request("PUT", url, headers=headers, data=payload)
    tt = response.json()
    print(tt,type(tt))
    self.broj_mogucih_akcija = 5  # Ovde postavite broj mogućih akcija
    self.action_space = Discrete(self.broj_mogucih_akcija)
    # Inicijalizacija okruženja
    self.sirina_slike = 3302
    self.visina_slike = 1651
    self.broj_kanala = 3  # 3 za RGB, 1 za grayscale
    self.sirina_resized = 64
    self.visina_resized = 32
    self.prev_gold = 0
    self.prev_value = 0
    self.observation_space = Box(low=0, high=255, shape=(self.visina_resized, self.sirina_resized, self.broj_kanala), dtype=np.uint8)
    self.trenutno_stanje = None
    self.reset()

######################
  def citaj_matricu_i_spoji_slike(self,matrica, mapa_simbola):
    visina, sirina = len(matrica), len(matrica[0])

    # Provera da li je mapa_simbola pravilno definisana
    if not all(isinstance(mapa_simbola[simbol], Image.Image) for red in matrica for simbol in red):
        raise ValueError("Svi simboli u mapi treba da budu objekti tipa PIL.Image.Image.")

    # Određivanje dimenzija rezultujuće slike
    velicina_slike = mapa_simbola[matrica[0][0]].size
    ukupna_visina = visina * velicina_slike[1]
    ukupna_sirina = sirina * velicina_slike[0]
    #print(ukupna_sirina, ukupna_visina, sep="\n\n")
    # Kreiranje rezultujuće slike
    rezultatna_slika = Image.new("RGB", (ukupna_sirina, ukupna_visina), color=(255, 255, 255))
    # Spajanje ikonica u rezultujuću sliku
    for i in range(visina):
        for j in range(sirina):
            simbol = matrica[i][j]
            ikonica = mapa_simbola[simbol]
            x_pozicija = j * velicina_slike[0]
            y_pozicija = i * velicina_slike[1]
            rezultatna_slika.paste(ikonica, (x_pozicija, y_pozicija))

    return rezultatna_slika
##########################

  def reset(self,number = 1):
    if number == -1:
      url = self.url_root+"/map/restart"
    else:
      url = self.url_root+f"/map/restart?map_number={number}"
    payload={}
    headers = {}
    response = requests.request("PUT", url, headers=headers, data=payload)
    output = response.json()

    url = self.url_root + "/map/full/matrix"
    response = requests.request("GET", url, headers=headers, data=payload)
    observationn = response.json()
    mapa_simbolaa = self.mapa_simbola
    originalna_slika = self.citaj_matricu_i_spoji_slike(observationn, mapa_simbolaa)  # Prilagodite putanju
    resized_slika = originalna_slika.resize((self.sirina_resized, self.visina_resized), Image.Resampling.LANCZOS)
    self.trenutno_stanje = np.array(resized_slika) 
    #print("Dimenzije trenutne slike nakon resetovanja:", self.trenutno_stanje["slika"].shape, sep="\n")
    return self.trenutno_stanje
    
##########################
  def step(self,action):
    url_sufix = "wait"
    if action == 0:
      url_sufix = "up"
    elif action == 1:
      url_sufix = "down"
    elif action == 2:
      url_sufix = "left"
    elif action == 3:
      url_sufix = "right"
    elif action == 4:
      url_sufix = "wait"

    url = self.url_root + "/player/" + url_sufix
    payload={}
    headers = {}
    response = requests.request("PUT", url, headers=headers, data=payload)
    time.sleep(0.02)

    url = self.url_root + "/map/full/matrix"
    response = requests.request("GET", url, headers=headers, data=payload)
    next_observation = response.json()

    url = self.url_root + "/player/inventory/value"
    response = requests.request("GET", url, headers=headers, data=payload)
    curr_value = response.json()

    url = self.url_root + "/player/inventory/gold"
    response = requests.request("GET", url, headers=headers, data=payload)
    curr_gold = response.json()

    url = self.url_root + "/map/isover"
    response = requests.request("GET", url, headers=headers, data=payload)
    done = response.json()

    reward = 0
    #if curr_value > self.prev_value:
    #        reward = 5
    if curr_value < self.prev_value:
            reward = 0 - 10
    if curr_gold > self.prev_gold:
            reward = 50
    if curr_gold < self.prev_gold:
            reward = 100
    
    mapa_simbolaa = self.mapa_simbola
    
    nova_opservacija = self.citaj_matricu_i_spoji_slike(next_observation, mapa_simbolaa)
    resized_slika = nova_opservacija.resize((self.sirina_resized, self.visina_resized), Image.Resampling.LANCZOS)
    inf = {}  
    self.trenutno_stanje = np.array(resized_slika)
    self.prev_gold = curr_gold
    self.prev_value = curr_value
    
    return self.trenutno_stanje, reward, done, inf

######################################
  def render(self):
    payload={}
    headers = {}

    url = self.url_root + "/map/full/matrix"
    response = requests.request("GET", url, headers=headers, data=payload)
    next_observation = response.json()

    mapa_simbolaa = self.mapa_simbola
    nova_opservacija = self.citaj_matricu_i_spoji_slike(next_observation, mapa_simbolaa)
    resized_slika = nova_opservacija.resize((self.sirina_resized, self.visina_resized), Image.Resampling.LANCZOS)
    self.trenutno_stanje = np.array(resized_slika)
    #self.trenutno_stanje["slika"] = np.array(resized_slika)

    return self.trenutno_stanje
  
##########################################
"""
class CustomGrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(CustomGrayScaleObservation, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.sirina_resized, self.visina_resized, 1), dtype=np.uint8)
    def observation(self, observation):
        # Implementirajte logiku za konverziju u nijanse sive boje
        gray_observation = ...
        return gray_observation
""" 
env = RafRpg()
#env = CustomGrayScaleObservation(env)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')
"""
try:
    state = env.reset()
except ValueError as e:
    print("Error during environment reset:", e)
state, reward, done, info = env.step([1])
state, reward, done, info = env.step([1])
state, reward, done, info = env.step([1])
state, reward, done, info = env.step([1])
plt.figure(figsize=(20,16))
num_channels = state.shape[-1]
# Prikazati najviše prvih 4 kanala
for idx in range(min(4, num_channels)):
    plt.subplot(1, min(4, num_channels), idx + 1)
    plt.imshow(state[0][:, :, idx])

plt.show()
"""
import os 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
callback = TrainAndLoggingCallback(check_freq=16000, save_path=CHECKPOINT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
"""
class CustomCNNPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNNPolicy, self).__init__(observation_space, features_dim)
        # Dobijanje dimenzija slike iz observation_space
        n_channels, height, width = observation_space.shape
        # Konvolucioni slojevi
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Potpuno povezani sloj
        self.fc1 = nn.Linear(self._get_conv_output_dim(n_channels, height, width), 512)
    def forward(self, x):
        x = x / 255.0  # Normalizacija slike na [0, 1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)  # Ravnanje za potpuno povezani sloj
        x = F.relu(self.fc1(x))
        return x
    def _get_conv_output_dim(self, n_channels, height, width):
        # Pomoćna metoda za određivanje dimenzije izlaza nakon konvolucije
        test_tensor = torch.ones(1, n_channels, height, width)
        test_output = self.conv2(self.conv1(test_tensor))
        return int(np.prod(test_output.size()))
# Kreiranje okruženja i politike
policy = CustomCNNPolicy(env.observation_space)
"""
"""
# Kreiranje PPO modela 
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, 
            learning_rate=0.000001,n_steps=256)#512
#TRENIRANJE
model.learn(total_timesteps=100000, callback=callback)
"""
"""
################
new_model = PPO.load('./train/best_model_66000')
# Nastavite treniranje novog modela
new_model.set_env(env)
new_model.learn(total_timesteps=50000, callback=callback) 
################
"""

#TESTIRANJE
model = PPO.load('./train/best_model_82000')
state = env.reset()
while True: 
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()

