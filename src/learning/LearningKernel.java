package learning;
import java.io.IOException;
import java.lang.Thread.State;
import java.util.ArrayList;
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
	
	/***NeuralNet Parameters*****/	
	private static int numStateCategory = 6;
	private static int numInput = numStateCategory;
	private static int numHidden = LUTNeuralNet.getNumHidden();
	private static int numOutput = LUTNeuralNet.getNumoutput();	
	private static double learningRate_NN = LUTNeuralNet.getLearningRate(); // alpha
	private static double momentumRate = LUTNeuralNet.getMomentumRate();
	private static double lowerBound = -1.0;
	private static double upperBound = 1.0;
	private double [] maxQ = new double[Actions.NumRobotActions];
	private double [] minQ = new double[Actions.NumRobotActions];
	private static int numNerualNet = 7;
	
	private int[] currentStateArray = new int [numStateCategory];
	private int[]  newStateArray = new int [numStateCategory];
	private double currentActionOutput [] = new double [numNerualNet];
	private double currentQValue[] = new double [numNerualNet];
	private double newActionOutput[] = new double [numNerualNet];
	private double newQValue[] =  new double [numNerualNet];
	
	private double errorSignal;
	
	private ArrayList<NeuralNet> neuralNetworks = new ArrayList<NeuralNet>();
	
	public LearningKernel (LUT table) {
		this.lut = table;
		for(int act = 0; act<Actions.NumRobotActions;act++) {
			maxQ[act] = 5+LUTNeuralNet.findMax(getColumn(lut.getTable(),act));
			minQ[act] = -5+LUTNeuralNet.findMin(getColumn(lut.getTable(),act));
		} 
		
		setErrorSignal(0.0);

	}
	

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
	

	public int selectAction(int state) {
		double epsl = Math.random();
		int action = 0;
		if(epsl < explorationRate) {
			Random rand = new Random();
			action = rand.nextInt(Actions.NumRobotActions);
		}else {

			action = lut.getMaxQAction(state);
		}
		return action;
	}
	
	//Always select action based on current state, current state is the state before an action is executed in each iteration
	public int nn_selectAction() {
		double epsl = Math.random();
		int action = 0;
		double [] inputData = LUTNeuralNet.normalizeInputData(getCurrentStateArray());
		if(epsl < explorationRate) {
			Random rand = new Random();
			action = rand.nextInt(Actions.NumRobotActions);//Exploration Move
		}else {
			//Greedy Move
			for(NeuralNet theNet : neuralNetworks) {
				int act = theNet.getNetID();
				double currentNetOutput = theNet.outputFor(inputData)[0];
				double currentNetQValue = LUTNeuralNet.inverseMappingOutput(currentNetOutput, maxQ[act], minQ[act], upperBound, lowerBound);//Reverse map output to big scale
				int currentNetIndex = theNet.getNetID();
				setCurrentActionValue(currentNetOutput,currentNetIndex);//Probably wrong
				setCurrentQValue(currentNetQValue,currentNetIndex);
			}
			
			action = getMaxIndex(getCurrentQValues());			
		}
		return action;
	}
	
	public void nn_QLearn( int action, double reward) {
		//Need to make currentData Array and new Data array is set before calling this function
		double currentStateQValue = getCurrentQValues()[action] ;
		double [] newInputData = new double[numStateCategory];
		newInputData = LUTNeuralNet.normalizeInputData(getNewStateArray());
		for(NeuralNet theNet: neuralNetworks) {
			int act = theNet.getNetID();
			double tempOutput = theNet.outputFor(newInputData)[0];
			double tempQValue = LUTNeuralNet.inverseMappingOutput(tempOutput, maxQ[act], minQ[act], upperBound, lowerBound);
			setNewActionValue(tempOutput,theNet.getNetID());
			setNewQValue(tempQValue,theNet.getNetID());
		}//Update the NewActionValue and newQValues Arrays
		
		int maxNewStateActionIndex = getMaxIndex(getNewQValues());
		double maxNewQValue = getNewQValues()[maxNewStateActionIndex];
		double expectedQValue = currentStateQValue + LearningRate*(reward + DiscountRate *maxNewQValue -currentStateQValue); 
		double [] expectedOutput = new double[1];
		expectedOutput[0] = LUTNeuralNet.normalizeExpectedOutput(expectedQValue, maxQ[action], minQ[action], upperBound, lowerBound);
		NeuralNet learningNet = neuralNetworks.get(action);
		double [] currentInputData = LUTNeuralNet.normalizeInputData(getCurrentStateArray());
		learningNet.train(currentInputData, expectedOutput);
		double tempOutput2 = learningNet.outputFor(currentInputData)[0];
		double tempQValue2 = LUTNeuralNet.inverseMappingOutput(tempOutput2, maxQ[action], minQ[action], upperBound, lowerBound);
		setErrorSignal(Math.abs(currentStateQValue - tempQValue2));
	}
	
	public void initializeNeuralNetworks(){
		for(int i = 0; i < Actions.NumRobotActions; i++) {
			NeuralNet theNewNet = new NeuralNet(numInput,numHidden,numOutput,learningRate_NN,momentumRate,lowerBound,upperBound,i);
			neuralNetworks.add(theNewNet);
		}
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
	
	public ArrayList<NeuralNet> getNeuralNetworks(){
		return this.neuralNetworks;
	}
	
	public void setErrorSignal(double error){
		errorSignal = error;
	}
	
	public double getErrorSignal(){
		return errorSignal;
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
		double[] column = new double[LUT.numStates]; // 
	    for(int i=0; i<column.length; i++){
	       column[i] = array[i][index];
	    }
	    return column;
	}
	
}
