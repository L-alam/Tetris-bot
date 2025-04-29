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

    // Higher initial exploration probability for better discovery
    public static final double INITIAL_EXPLORATION_PROB = 0.8;
    public static final double MIN_EXPLORATION_PROB = 0.05;
    
    private Random random;
    private double currentExplorationProb;
    private long lastGameCount = 0;
    private int totalGamesPlayed = 0;
    
    // Store previous score to calculate improvements
    private int previousTotalScore = 0;

    public TetrisQAgent(String name) {
        super(name);
        this.random = new Random(12345);
        this.currentExplorationProb = INITIAL_EXPLORATION_PROB;
    }

    public Random getRandom() { 
        return this.random; 
    }

    @Override
    public Model initQFunction() {
        // Create a neural network with four layers for better pattern recognition
        final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        final int firstHiddenLayer = 512;  // Larger first layer
        final int secondHiddenLayer = 256; // Larger second layer
        final int thirdHiddenLayer = 128;  // Added third layer
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        
        // Input layer to first hidden layer
        qFunction.add(new Dense(numPixelsInImage, firstHiddenLayer));
        qFunction.add(new ReLU());
        
        // First hidden layer to second hidden layer
        qFunction.add(new Dense(firstHiddenLayer, secondHiddenLayer));
        qFunction.add(new ReLU());
        
        // Second hidden layer to third hidden layer
        qFunction.add(new Dense(secondHiddenLayer, thirdHiddenLayer));
        qFunction.add(new ReLU());
        
        // Third hidden layer to output layer
        qFunction.add(new Dense(thirdHiddenLayer, outDim));
        
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
        // More sophisticated exploration strategy with faster decay
        int currentCycleIdx = (int)gameCounter.getCurrentCycleIdx();
        int currentGameIdx = (int)gameCounter.getCurrentGameIdx();
        long totalGames = currentCycleIdx * 1000 + currentGameIdx;
        
        // Update exploration probability only when we've played more games
        if (totalGames > lastGameCount) {
            lastGameCount = totalGames;
            
            // Decay exploration probability based on total games played
            // This creates a smoother decay curve
            double decayFactor = Math.exp(-0.0001 * totalGames);
            currentExplorationProb = MIN_EXPLORATION_PROB + 
                                    (INITIAL_EXPLORATION_PROB - MIN_EXPLORATION_PROB) * decayFactor;
            
            totalGamesPlayed++;
        }
        
        // Add randomness to exploration based on game state
        if (game.getScoreThisTurn() > 0) {
            // Reduce exploration probability during good moves
            return this.getRandom().nextDouble() <= (currentExplorationProb * 0.5);
        }
        
        return this.getRandom().nextDouble() <= currentExplorationProb;
    }

    @Override
    public Mino getExplorationMove(final GameView game) {
        // Smarter exploration strategy
        List<Mino> possibleMoves = game.getFinalMinoPositions();
        
        // If we have very few moves, just pick one
        if (possibleMoves.size() <= 2) {
            return possibleMoves.get(0);
        }
        
        // Keep track of the best move we've found
        Mino bestMove = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        
        // Try each move and evaluate it heuristically
        for (Mino move : possibleMoves) {
            // Start with a random base score for diversity
            double moveScore = this.getRandom().nextDouble() * 10;
            
            // Check if the move scores points
            int potentialScore = game.getScoreThisTurn();
            if (potentialScore > 0) {
                moveScore += potentialScore * 2;
            }
            
            // If this is the best move we've seen, remember it
            if (moveScore > bestScore) {
                bestScore = moveScore;
                bestMove = move;
            }
        }
        
        // If we found a good move, use it
        if (bestMove != null) {
            return bestMove;
        }
        
        // Fallback to random selection
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
        
        // Store current total score
        int currentTotalScore = game.getTotalScore();
        
        // Calculate score improvement from last turn
        int scoreDifference = currentTotalScore - previousTotalScore;
        previousTotalScore = currentTotalScore;
        
        // Strongly reward scoring points
        if (scoreDifference > 0) {
            // Exponential reward based on score - greatly rewards higher scores
            reward += Math.pow(scoreDifference, 1.5);
        }
        
        // Add a small positive reward for each action to encourage continued play
        reward += 1.0;
        
        // Substantial penalty for game over
        if (game.isGameOver()) {
            reward -= 1000;
            // Reset previous score for next game
            previousTotalScore = 0;
        }
        
        return reward;
    }
}