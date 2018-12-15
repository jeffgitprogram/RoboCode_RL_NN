package learning;
import java.io.IOException;
import java.lang.Thread.State;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import Neurons.*;
public class LearningKernel {
	public static final double LearningRate = 0.3;   // alpha
	public static final double DiscountRate = 0.9;   // gamma
	public static double explorationRate = 0.2; 
	private int currentState;   
	private int currentAction;
	private boolean isFirstRound = true;
	private LUT lut; 
	private NeuralNet neuralNet;
	
	/***NeuralNet Parameters*****/	
	private static int numStateCategory = 6;
	private static int numInput = numStateCategory;
	private static int numHidden = LUTNeuralNet.getNumHidden();
	private static int numOutput = LUTNeuralNet.getNumoutput();	
	private static double learningRate_NN = LUTNeuralNet.getLearningRate(); // alpha
	private static double momentumRate = LUTNeuralNet.getMomentumRate();
	private static double lowerBound = -1.0;
	private static double upperBound = 1.0;
	private double maxQ;
	private double minQ;
	//private static int numNerualNet = 7;
	
	private int[] currentStateArray = new int [numStateCategory];
	private int[]  newStateArray = new int [numStateCategory];
	private double currentActionOutput [] = new double [Actions.NumRobotActions];
	private double currentQValue[] = new double [Actions.NumRobotActions];
	private double newActionOutput[] = new double [Actions.NumRobotActions];
	private double newQValue[] =  new double [Actions.NumRobotActions];
	
	//private ArrayList<NeuralNet> neuralNetworks = new ArrayList<NeuralNet>();
	
	public LearningKernel (LUT table) {
		this.lut = table;
		for(int act = 0; act<Actions.NumRobotActions;act++) {
			if(LUTNeuralNet.findMax(getColumn(lut.getTable(),act))> maxQ) 
			{
				maxQ = LUTNeuralNet.findMax(getColumn(lut.getTable(),act));
			}
			if(minQ > LUTNeuralNet.findMin(getColumn(lut.getTable(),act))) 
			{
				minQ = LUTNeuralNet.findMin(getColumn(lut.getTable(),act));
			}
		}//Find min and max of the whole LUT
		maxQ = (int)maxQ+5;
		minQ = (int)minQ-5;
		System.out.println("break here.");
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
	//Always select action based on current state, current state is the state before an action is executed in each iteration
	public int nn_selectAction() {
		double epsl = Math.random();
		int action = 0;
		double [] inputData = null;
		if(epsl < explorationRate) {
			Random rand = new Random();
			action = rand.nextInt(Actions.NumRobotActions);//Exploration Move
		}else {
			//Greedy Move
			int[] state_with_action = Arrays.copyOf(getCurrentStateArray(), getCurrentStateArray().length+1);
			for(int act = 0; act<Actions.NumRobotActions; act++) {
				state_with_action[state_with_action.length-1]= act;
				inputData = LUTNeuralNet.normalizeInputData(state_with_action);
				double currentNetOutput = neuralNet.outputFor(inputData)[0];
				double currentNetQValue = LUTNeuralNet.inverseMappingOutput(currentNetOutput, maxQ, minQ, upperBound, lowerBound);//Reverse map output to big scale
				setCurrentActionValue(currentNetOutput,act);//Probably wrong
				setCurrentQValue(currentNetQValue,act);
			}			
			action = getMaxIndex(getCurrentQValues());	
		}
		return action;
	}
	
	public void nn_QLearn( int action, double reward) {
		//Need to make currentData Array and new Data array is set before calling this function
		double currentStateQValue = getCurrentQValues()[action] ;
		double [] newInputData = null;
		int[] state_with_action = Arrays.copyOf(getNewStateArray(), getNewStateArray().length+1);		
		for(int act = 0; act<Actions.NumRobotActions; act++) {
			state_with_action[state_with_action.length-1]= act;//Combine the state input with a selected act
			newInputData = LUTNeuralNet.normalizeInputData(state_with_action);
			double tempOutput = neuralNet.outputFor(newInputData)[0];
			double tempQValue = LUTNeuralNet.inverseMappingOutput(tempOutput, maxQ, minQ, upperBound, lowerBound);
			setNewActionValue(tempOutput,act);
			setNewQValue(tempQValue,act);
		}//Update the NewActionValue and newQValues Arrays
		
		int maxNewStateActionIndex = getMaxIndex(getNewQValues());
		double maxNewQValue = getNewQValues()[maxNewStateActionIndex];
		double expectedQValue = currentStateQValue + LearningRate*(reward + DiscountRate *maxNewQValue -currentStateQValue); 
		double [] expectedOutput = new double[1];
		expectedOutput[0] = LUTNeuralNet.normalizeExpectedOutput(expectedQValue, maxQ, minQ, upperBound, lowerBound);
		int[] current_State_Action = Arrays.copyOf(getCurrentStateArray(), getCurrentStateArray().length+1);
		current_State_Action[current_State_Action.length-1] = action;
		double [] currentInputData = LUTNeuralNet.normalizeInputData(current_State_Action);
		neuralNet.train(currentInputData, expectedOutput);
	}
	
	public void initializeNeuralNetworks(){
		neuralNet = new NeuralNet(numInput,numHidden,numOutput,learningRate_NN,momentumRate,lowerBound,upperBound,0);
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
	
	public void setCurrentActionValues(double [] theValues) {
		currentActionOutput = theValues;
	}
	public void setCurrentActionValue(double theValues, int theIndex) {
		currentActionOutput[theIndex] = theValues;
	}
	public double [] getCurrentActionValues() {
		return this.currentActionOutput;
	}
	
	public void setNewActionValues(double [] theValues) {
		newActionOutput = theValues;
	}
	public void setNewActionValue(double theValues, int theIndex) {
		newActionOutput[theIndex] = theValues;
	}
	public double [] getNewActionValues() {
		return this.newActionOutput;
	}
	public void setCurrentQValues(double [] theValues) {
		currentQValue = theValues;
	}
	public void setCurrentQValue(double theValues, int theIndex) {
		currentQValue[theIndex] = theValues;
	}
	public double [] getCurrentQValues() {
		return this.currentQValue;
	}
	
	public void setNewQValues(double [] theValues) {
		newQValue = theValues;
	}
	public void setNewQValue(double theValues, int theIndex) {
		newQValue[theIndex] = theValues;
	}
	public double [] getNewQValues() {
		return this.newQValue;
	}
	
	public NeuralNet getNeuralNetwork(){
		return this.neuralNet;
	}
	
	public int getMaxIndex(double [] theValues) {
		double maxQValue = theValues[0];
		int maxIndex = 0;
		for(int i = 0; i < theValues.length; i++) {
			if(maxQValue < theValues[i]) {
				maxQValue = theValues[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	} 
	
	public double[] getColumn(double[][] array, int index){
		double[] column = new double[States.NumStates]; // 
	    for(int i=0; i<column.length; i++){
	       column[i] = array[i][index];
	    }
	    return column;
	}
	
}
