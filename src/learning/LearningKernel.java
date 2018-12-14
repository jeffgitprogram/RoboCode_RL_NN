package learning;
import java.lang.Thread.State;
import java.util.ArrayList;
import java.util.Random;

import Neurons.NeuralNet;
public class LearningKernel {
	public static final double LearningRate = 0.3;   // alpha
	public static final double DiscountRate = 0.9;   // gamma
	public static double explorationRate = 0.2; 
	private int currentState;   
	private int currentAction;
	private boolean isFirstRound = true;
	private LUT lut; 
	
	/***NeuralNet Parameters*****/	
	private static int numStateCategory = 6;
	private static int numInput = numStateCategory;
	private static int numHidden = 40;
	private static int numOutput = 1;	
	private static double learningRate_NN = 0.005; // alpha
	private static double momentumRate = 0.9;
	private static double lowerBound = -1.0;
	private static double upperBound = 1.0;
	private static double maxQ = 120;
	private static double minQ = -20;
	private static int numNerualNet = 7;
	
	private int[] currentStateArray = new int [numStateCategory];
	private int[]  newStateArray = new int [numStateCategory];
	private double currentActionOutput [] = new double [numNerualNet];
	private double currentQValue[] = new double [numNerualNet];
	private double newActionOutput[] = new double [numNerualNet];
	private double newQValue[] =  new double [numNerualNet];
	
	private static ArrayList<NeuralNet> neuralNetworks = new ArrayList<NeuralNet>();
	
	public LearningKernel (LUT table) {
		this.lut = table;
	}
	
	//Off-policy learning
	public void QLearn (int nextState, int nextAction, double reward) {
		double lastQVal;
		double newQVal;
		if(isFirstRound) {
			isFirstRound = false;			
		}
		else {
			lastQVal = lut.getQValue(currentState, currentAction);
			newQVal  = lastQVal + LearningRate*(reward + DiscountRate * lut.getMaxQvalue(nextState)-lastQVal);
			lut.setQvalue(currentState, currentAction, newQVal);
		}
		
		currentState = nextState;
		currentAction = nextAction;
	}
	
	//On-policy Learning
	public void SARSLearn(int nextState, int nextAction, double reward) {
		double lastQVal;
		double newQVal;
		if(isFirstRound) {
			isFirstRound = false;			
		}
		else {
			lastQVal = lut.getQValue(currentState, currentAction);
			newQVal = lastQVal + LearningRate*(reward + DiscountRate * lut.getQValue(nextState, nextAction) - lastQVal);
			lut.setQvalue(currentState, currentAction, newQVal);
		}
		
		currentState = nextState;
		currentAction = nextAction;
	}
	
	//Episilon-Greedy
	public int selectAction(int state) {
		double epsl = Math.random();
		int action = 0;
		if(epsl < explorationRate) {
			Random rand = new Random();
			action = rand.nextInt(Actions.NumRobotActions);//Exploration Move
		}else {
			//Greedy Move
			action = lut.getMaxQAction(state);
		}
		return action;
	}
	
	public void setCurrentStateArray (int state) {
		currentStateArray = States.getStateFromIndex(state);
	}
	
	public int [] getCurrentStateArray(){
		return this.currentStateArray;
	}
	public void setNewStateArray (int state) {
		newStateArray = States.getStateFromIndex(state);
	}
	
	public int [] getNewStateArray(){
		return this.newStateArray;
	}
	
	
}
