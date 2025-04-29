package src.pas.tetris.agents;

// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;

// JAVA PROJECT IMPORTS
import edu.bu.pas.tetris.agents.QAgent;
import edu.bu.pas.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.pas.tetris.game.Board;
import edu.bu.pas.tetris.game.Game.GameView;
import edu.bu.pas.tetris.game.minos.Mino;
import edu.bu.pas.tetris.linalg.Matrix;
import edu.bu.pas.tetris.nn.Model;
import edu.bu.pas.tetris.nn.LossFunction;
import edu.bu.pas.tetris.nn.Optimizer;
import edu.bu.pas.tetris.nn.models.Sequential;
import edu.bu.pas.tetris.nn.layers.Dense;
import edu.bu.pas.tetris.nn.layers.ReLU;
import edu.bu.pas.tetris.nn.layers.Tanh;
import edu.bu.pas.tetris.nn.layers.Sigmoid;
import edu.bu.pas.tetris.training.data.Dataset;
import edu.bu.pas.tetris.utils.Pair;

public class TetrisQAgent extends QAgent {

    // Use a static exploration probability with the option to adjust later
    public static final double EXPLORATION_PROB = 0.1;
    
    private Random random;
    private int gamesPlayed = 0;

    public TetrisQAgent(String name) {
        super(name);
        this.random = new Random(12345);
    }

    public Random getRandom() { 
        return this.random; 
    }

    @Override
    public Model initQFunction() {
        // Create a neural network with three hidden layers
        final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        final int firstHiddenLayer = 256;
        final int secondHiddenLayer = 128;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        
        // Input layer to first hidden layer
        qFunction.add(new Dense(numPixelsInImage, firstHiddenLayer));
        qFunction.add(new ReLU());
        
        // First hidden layer to second hidden layer
        qFunction.add(new Dense(firstHiddenLayer, secondHiddenLayer));
        qFunction.add(new ReLU());
        
        // Second hidden layer to output layer
        qFunction.add(new Dense(secondHiddenLayer, outDim));
        
        return qFunction;
    }

    @Override
    public Matrix getQFunctionInput(final GameView game, final Mino potentialAction) {
        // Get the grayscale image representation
        Matrix input = null;
        try {
            input = game.getGrayscaleImage(potentialAction).flatten();
        } catch(Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        
        return input;
    }

    @Override
    public boolean shouldExplore(final GameView game, final GameCounter gameCounter) {
        // Simpler exploration strategy based on fixed probability
        // We track the cycle index as an int to avoid the type conversion error
        int currentCycleIdx = (int)gameCounter.getCurrentCycleIdx();
        
        // Decrease exploration probability as training progresses
        double exploreProb = EXPLORATION_PROB;
        if (currentCycleIdx > 100) {
            exploreProb *= 0.9; // Reduce by 10% after 100 cycles
        }
        if (currentCycleIdx > 500) {
            exploreProb *= 0.9; // Reduce by another 10% after 500 cycles
        }
        if (currentCycleIdx > 1000) {
            exploreProb *= 0.8; // Reduce by another 20% after 1000 cycles
        }
        
        return this.getRandom().nextDouble() <= exploreProb;
    }

    @Override
    public Mino getExplorationMove(final GameView game) {
        // Get all possible final positions
        List<Mino> possibleMoves = game.getFinalMinoPositions();
        
        // Simple heuristic - select randomly with slight preference for moves that might clear lines
        for (Mino move : possibleMoves) {
            // Try to find moves that are likely to clear lines
            // Since we don't have access to getCompletedLines or similar methods,
            // we'll use a simpler approach based on random selection
            if (this.getRandom().nextDouble() < 0.2) {
                return move;
            }
        }
        
        // If no move was selected by the heuristic, choose randomly
        int randIdx = this.getRandom().nextInt(possibleMoves.size());
        return possibleMoves.get(randIdx);
    }

    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    @Override
    public double getReward(final GameView game) {
        double reward = 0.0;
        
        // Reward for scoring points - this is the main signal available
        reward += game.getScoreThisTurn();
        
        // Add a larger reward multiplier for scores to make the signal stronger
        if (game.getScoreThisTurn() > 0) {
            reward += game.getScoreThisTurn() * 2;
        }
        
        reward += 0.1;
        
        return reward;
    }
}