package src.pas.tetris.agents;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Block;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Coordinate;
import edu.bu.tetris.utils.Pair;


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        // System.out.println("initQFunction called!");
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector

        // build a 4 layer network with ReLU activation functions, with 
        // the first hidden layer having size 30 and second 15 
        final int inDim = 15;
        final int hiddenDim1 = 2 * inDim;
        final int hiddenDim2 = hiddenDim1 / 2;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inDim, hiddenDim1));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim1, hiddenDim2));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim2, outDim));

        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        Matrix inputVector = Matrix.full(1, 15, 0.0);
        try {
            Matrix board = game.getGrayscaleImage(potentialAction);

            // get all of the feature values
            List<Double> columnHeights = this.getColumnHeights(board);
            double stackHeight = this.getStackHeight(columnHeights);
            double heightVariance = this.getHeightVariance(columnHeights);
            double bumpiness = this.getBumpiness(columnHeights);
            double numClearedLines = this.getClearedLines(board);
            double numHoles = this.getNumHoles(board);

            // normalize and add all of the feature values to the Q function input vector
            for (int i = 0; i < Board.NUM_COLS; i++) {
                inputVector.set(0, i, columnHeights.get(i)/Board.NUM_ROWS);
            }
            inputVector.set(0, 10, stackHeight/(Board.NUM_ROWS * Board.NUM_COLS));
            inputVector.set(0, 11, heightVariance/Board.NUM_ROWS);
            inputVector.set(0, 12, bumpiness/((Board.NUM_COLS - 1) * Board.NUM_ROWS));
            inputVector.set(0, 13, numClearedLines/Board.NUM_ROWS);
            inputVector.set(0, 14, numHoles/(Board.NUM_ROWS * Board.NUM_COLS));
        } catch(Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return inputVector;
    }

    /* Get the heights of each column and return them as a list, given the Board object. */
    private List<Double> getColumnHeights(Board board) {
        List<Double> heights = new ArrayList<>();

        for (int c = 0; c < Board.NUM_COLS; c++) {
            double colHeight =  0.0;
            for (int r = 0; r < Board.NUM_ROWS; r++) {
                if (board.isCoordinateOccupied(c, r)) {
                    colHeight = Board.NUM_ROWS - r;
                    break;
                }
            }

            heights.add(colHeight);
        }
        
        return heights;
    }

    /* Get the heights of each column and return them as a list, given a matrix representation of the board. */
    private List<Double> getColumnHeights(Matrix board) {
        List<Double> heights = new ArrayList<>();
        
        for (int c = 0; c < Board.NUM_COLS; c++) {
            double colHeight =  0.0;
            for (int r = 0; r < Board.NUM_ROWS; r++) {
                if (board.get(r, c) > 0.0) {
                    colHeight = Board.NUM_ROWS - r;
                    break;
                }
            }

            heights.add(colHeight);
        }
        
        return heights;
    }

    /* Get the total height of the stack by summing all of the column heights */
    private double getStackHeight(List<Double> columnHeights) {
        double sumHeights = 0;
        for(Double d : columnHeights) {
            sumHeights += d;
        }
        return sumHeights;
    }

    /* Get the variance in the heights of the columns in the board, measured simply as the difference between the
     * max column height and the min column height
     */
    private double getHeightVariance(List<Double> columnHeights) {
        return (Collections.max(columnHeights) - Collections.min(columnHeights));
    }

    /* Get the bumpiness of the board, measured by the sum of the differences in heights between adjacent columns */
    private double getBumpiness (List<Double> columnHeights) {
        double bumpiness = 0.0;

        for (int i = 0; i < columnHeights.size() - 1; i++) {
            bumpiness += Math.abs(columnHeights.get(i) - columnHeights.get(i+1));
        }

        return bumpiness;
    }

    /* Determine the number of lines that would be cleared if we did the potential action (put the Mino in that spot) */
    private double getClearedLines (Matrix board) {
        double numClearedLines = 0;
 
        for (int r = 0; r < Board.NUM_ROWS; r++) {
            // Check that each column in this row is filled, set rowIsClear to false and break
            // once we encounter an empty cell
            boolean rowIsClear = true;
            for (int c = 0; c < Board.NUM_COLS; c++) {
                if (board.get(r, c) == 0.0) {
                    rowIsClear = false;
                    break;
                }
            }

            if (rowIsClear) {
                numClearedLines++;
            }
        }

        return numClearedLines;
    }


    /* Count the number of holes in the board, where holes are empty cells with an occupied cell directly above,
     * given the Board object. 
     */
    private double getNumHoles(Board board) {
        double numHoles = 0;

        for (int c = 0; c < Board.NUM_COLS; c++) {
            boolean filledSpaceAbove = false;
            for (int r = 1; r < Board.NUM_ROWS; r++) {
                if (board.isCoordinateOccupied(c, r)) { 
                    filledSpaceAbove = true;
                }

                if (!board.isCoordinateOccupied(c, r) && filledSpaceAbove) {
                    numHoles++;
                }
            }
        }

        return numHoles;
    }

    /* Count the number of holes in the board, where holes are empty cells with an occupied cell directly above,
     * given a Matrix representation of the board. 
     */
    private double getNumHoles(Matrix board) {
        double numHoles = 0;

        for (int c = 0; c < Board.NUM_COLS; c++) {
            boolean filledSpaceAbove = false;
            for (int r = 1; r < Board.NUM_ROWS; r++) {
                if (board.get(r, c) > 0.0) { 
                    filledSpaceAbove = true;
                }

                if (board.get(r, c) == 0.0 && filledSpaceAbove) {
                    numHoles++;
                }
            }
        }

        return numHoles;
    }

    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        // System.out.println("phaseIdx=" + gameCounter.getCurrentPhaseIdx() + "\tgameIdx=" + gameCounter.getCurrentGameIdx());

        // start with high probability of exploration, decrease it over time
        long currentGameIdx = gameCounter.getCurrentGameIdx();
        double initialProb = 1.0;
        double decayRate = 0.0001;
        double minProb = 0.05;

        double exploreProb = Math.max(minProb, initialProb * Math.exp(-decayRate * currentGameIdx));

        return this.getRandom().nextDouble() <= exploreProb;
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        List<Mino> potentialActions = game.getFinalMinoPositions();
        double[] rewardSamples = new double[potentialActions.size()];

        // loop through all potential actions and calculate their estimated rewards with the features from the potential boards
        for (int i = 0; i < potentialActions.size(); i++) {
            try {
                Mino potentialAction = potentialActions.get(i);
                Matrix potentialBoard = game.getGrayscaleImage(potentialAction);

                List<Double> columnHeights = this.getColumnHeights(potentialBoard);
                double stackHeight = this.getStackHeight(columnHeights);
                double heightVariance = this.getHeightVariance(columnHeights);
                double bumpiness = this.getBumpiness(columnHeights);
                double numClearedLines = this.getClearedLines(potentialBoard);
                double numHoles = this.getNumHoles(potentialBoard);

                double rewardEstimation = 7*(numClearedLines) - 5*(stackHeight)/(Board.NUM_ROWS * Board.NUM_COLS) - 4*(numHoles/(Board.NUM_ROWS * Board.NUM_COLS))
                                - 2.5*(bumpiness/((Board.NUM_COLS - 1) * Board.NUM_ROWS)) - 1*(heightVariance/Board.NUM_ROWS);
                rewardSamples[i] = rewardEstimation;
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(-1);
            }
        }

        // get weighted probabilities from the calculated rewards by softmaxing
        double[] probabilities = new double[potentialActions.size()];
        double max = Arrays.stream(rewardSamples).max().getAsDouble(); 
        double sum = 0;
        for (int i = 0; i < rewardSamples.length; i++) {
            probabilities[i] = Math.exp(rewardSamples[i] - max); 
            sum += probabilities[i];
        }
        for (int i = 0; i < rewardSamples.length; i++) {
            probabilities[i] /= sum;
        }

        // pick an index for an exploration move with the weighted probabilities
        double rand = this.getRandom().nextDouble();
        double cumulativeProb = 0;
        int sampleIdx = rewardSamples.length - 1;
        for (int i = 0; i < probabilities.length; i++) {
            cumulativeProb += probabilities[i];
            if (rand < cumulativeProb) {
                sampleIdx = i;
                break;
            }
        }

        return potentialActions.get(sampleIdx);
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
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

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game)
    {
        double reward = 0.0;

        try {
            Board board = game.getBoard();

            // get feature values
            double pointsThisTurn = game.getScoreThisTurn();
            List<Double> columnHeights = this.getColumnHeights(board);
            double stackHeight = this.getStackHeight(columnHeights);
            double heightVariance = this.getHeightVariance(columnHeights);
            double bumpiness = this.getBumpiness(columnHeights);
            double numHoles = this.getNumHoles(board);
            
            // reward function
            reward = 7*(pointsThisTurn) - 5*(stackHeight)/(Board.NUM_ROWS * Board.NUM_COLS) - 4*(numHoles/(Board.NUM_ROWS * Board.NUM_COLS)) 
                    - 2.5*(bumpiness/((Board.NUM_COLS - 1) * Board.NUM_ROWS)) - 1*(heightVariance/Board.NUM_ROWS);
        
            if (game.didAgentLose()) {
                reward -= 500 ; // negative reward for terminal state
            }
        } catch(Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        
        return reward;
    }

}
