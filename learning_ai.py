import numpy as np

class Player(object):
    '''
    Parent Class for all player types (different AIs).
    Simply defines some consistent things for each player.
    Each Player type inherits from this object.
    '''
    def __init__(self, p=1, name='Player'):
        self.player  = p      # Player Number 
        self.name    = name   # Player Name (for display purposes only)
        self.marker  = 1 if self.player == 1 else 2 # Marker / token to be displayed on the Grid
        self.target  = 4*self.marker # Target value to flag when player won the game

class LearningAI(Player):
    
    '''
    Player type that uses a Keras CNN model to make decisions.
    
    Model input :
    6x7 numpy array representing the game grid, or list of such grids, but the model has to 
    be reshaped to (N_grids, 6, 7, 1), adding an extra dimension to mimic an image.
    
    Model Output Layer :
    1 Single node with Sigmoid Activation function (mean't to represent the likelihood of winning given a
    certain grid).  
    
    Important note : for now the implementation assumes that the LearningAI is Player 1.  Because of this
    it makes decisions based on the assumption that it wants the 1 marker to win on the game grid.
    '''
        
    # Constructor
    def __init__(self,
                    keras_model, # Path to keras Conv2D model. 
                    p=1,
                    name="Paul"):

        import keras.models as km # Do it here so that we don't have to if this AI isn't playing
        
        Player.__init__(self, p, name)        # Parent class declarations
        self.model = keras_model # Load Keras Model
        self.player_type = 'LearningAI'       # Object name (used when need arises)


    def move(self, Board):
        '''
        Assigns column choice to .choice attribute.
        '''
        
        # Get Available column and corresponding row indices
        self.col_indices = [i for i,v in enumerate(Board.col_moves) if v != 0]
        row_indices      = [i - 1 for i in Board.col_moves if i != 0]

        # Make array of potential board states, each with the players next possible moves
        potential_states = np.array([Board.grid.copy() for i in self.col_indices])
        
        for i, v in enumerate(self.col_indices):
            potential_states[i][row_indices[i]][v] = self.marker

        # Reshape potential states so that it fits into the Conv2D model
        potential_states = np.array([s.reshape(6,7,1) for s in potential_states])

        # Make predictions with Model object
        self.predictions = self.model.predict(potential_states).flatten()


        # Select prediction closest to 1 (likelihood of winning?)
        # and assign it to the choice attribute
        best_move = np.where(self.predictions == self.predictions.max())[0][0]
        self.choice = self.col_indices[best_move]

        return 0

    def print_move_weights(self):
        '''
        Print the predictions of the different model predictions and the max value. 
        Mostly used to check that things were working.
        '''
        tuples = [(self.col_indices[i],v) for i,v in enumerate(self.predictions)]
        print("Move Weights :", tuples)


        pass

