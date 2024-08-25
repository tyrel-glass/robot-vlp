import numpy as np


class Robot:
  def __init__(self, x, y, heading, step_err, turn_err,df, vlp_mod, target_x= 0 , target_y = 0 , navigation_method = 'odometer', nav_model = None ):
    self.x = x
    self.vlp_x = x
    self.model_x = 0
    self.y = y
    self.vlp_y = y
    self.model_y = 0
    self.heading = heading
    self.vlp_heading = heading
    self.model_heading = 0
    self.step_err = step_err
    self.turn_err = turn_err
    self.df = df
    self.vlp_mod = vlp_mod

    self.navigation_method = navigation_method

    self.nav_model = nav_model

    self.x_hist = [x]
    self.y_hist = [y]
    self.heading_hist = [heading]

    self.vlp_x_hist = [self.vlp_x]
    self.vlp_y_hist = [self.vlp_y]

    self.model_x_hist = [self.model_x]
    self.model_y_hist = [self.model_y]
    self.model_heading_hist = [0]
    self.get_vlp_position()
    self.vlp_heading_hist = [self.vlp_heading] # make random

    self.encoder_heading = heading
    self.encoder_heading_hist = [heading]
    self.encoder_x = x
    self.encoder_y = y
    self.encoder_x_hist = [self.encoder_x]
    self.encoder_y_hist = [self.encoder_y]

    self.target_x = target_x
    self.target_y = target_y

  def turn(self, angle = 0):
    
    heading_error = self.turn_err * angle *(np.random.random() - 0.5)*2   # error = +/- turn_err % (of stepsize)
    self.encoder_heading = self.encoder_heading + angle 
    self.heading = self.heading + angle + heading_error
    
    
    # self.record_state()

  def step(self, step_size = 1):
    forward_error = self.step_err * step_size *(np.random.random()-0.5)*2  # error = +/- step_err % (of stepsize)
    encoder_step_size = step_size + forward_error

    self.x = self.x + np.sin(self.heading*np.pi/180) * encoder_step_size
    self.y = self.y + np.cos(self.heading*np.pi/180) * encoder_step_size

    

    self.encoder_x = self.encoder_x + np.sin(self.encoder_heading*np.pi/180) * step_size
    self.encoder_y = self.encoder_y + np.cos(self.encoder_heading*np.pi/180) * step_size

    self.record_state()


  def record_state(self):
    self.x_hist.append(self.x)
    self.y_hist.append(self.y)
    
    self.get_vlp_position()
    self.vlp_x_hist.append(self.vlp_x)
    self.vlp_y_hist.append(self.vlp_y)
    self.get_vlp_heading()
    self.vlp_heading_hist.append(self.vlp_heading)

    self.heading_hist.append(self.heading)

    self.encoder_y_hist.append(self.encoder_y)
    self.encoder_x_hist.append(self.encoder_x)
    self.encoder_heading_hist.append(self.encoder_heading)


  def get_vlp_position(self):
    self.vlp_x, self.vlp_y = get_vlp_pos_estimate(self.df, self.x, self.y, self.vlp_mod)


  def get_vlp_heading(self):
    if len(self.vlp_x_hist)>1:
      x_d =  self.vlp_x_hist[-1] - self.vlp_x_hist[-2]
      y_d =  self.vlp_y_hist[-1] - self.vlp_y_hist[-2],

      new_heading = np.arctan2(x_d, y_d)[0]*180/np.pi
    else:
      new_heading = 0
    self.vlp_heading = new_heading
    

  def get_model_update(self):
    self.model_x = None
    self.model_y = None
    self.model_heading = None


  def calc_heading_to_target(self):
    if self.navigation_method == 'odometer':
      x_d = self.target_x - self.encoder_x
      y_d = self.target_y - self.encoder_y

    elif self.navigation_method == 'vlp':
      x_d = self.target_x - self.vlp_x
      y_d = self.target_y - self.vlp_y
    elif self.navigation_method == 'model':
      x_d = self.target_x - self.model_x
      y_d = self.target_y - self.model_y

    ang_to_tar = np.arctan2(x_d, y_d)*180/np.pi
    return ang_to_tar
  
  def correct_heading(self):

    ang_to_tar = self.calc_heading_to_target()
    if self.navigation_method == 'odometer':
      ang_corr = ang_to_tar - self.encoder_heading
    elif self.navigation_method == 'vlp':
      ang_corr = ang_to_tar - self.encoder_heading
    elif self.navigation_method == 'model':
      ang_corr = ang_to_tar - self.model_heading
      pass # need to write in here
    
    self.turn(ang_corr)

  def calc_distance_to_target(self):
    if self.navigation_method == 'odometer':
      x_e = self.encoder_x
      y_e = self.encoder_y
    elif self.navigation_method == 'vlp':
      x_e = self.vlp_x
      y_e = self.vlp_y
    elif self.navigation_method == 'model':
      x_e = self.model_x
      y_e = self.model_y 

    x_t = self.target_x
    y_t = self.target_y

    tar_dist = np.sqrt((x_t-x_e)**2 + (y_t - y_e)**2)
    return tar_dist




def get_vlp_pos_estimate(df,x,y,vlp_mod):
    closest_index = find_closest_index(df, x, y)

    vlp_sigs = df.iloc[closest_index:closest_index + 1, :11].values
    position_prediction = vlp_mod.predict(vlp_sigs)[0]
    estimated_x = position_prediction[0]
    estimated_y = position_prediction[1]
    estimated_x, estimated_y
    return estimated_x, estimated_y

def find_closest_index(df, x, y):
    return np.argmin(np.square(df['x'] - x) + np.square(df['y'] - y))
