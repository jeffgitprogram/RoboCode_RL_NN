package learning;

public class States {
	public static final int NumHeading = 4;  //Four states, up, right, down, left
	public static final int NumTargetDistance = 10;  //Ten levels of distance
	public static final int NumTargetBearing = 4;  
	//public static final int NumHitWall = 2;  
	public static final int NumHorizontalPositionUnsafe = 2; 
	public static final int NumVerticalPositionUnsafe = 2; 
	public static final int NumHitByBullet = 2;  
	public static final int NumStates;  
	public static final int Mapping[][][][][][];
	
	static  {  
		Mapping = new int[NumHeading][NumTargetDistance][NumTargetBearing][NumHorizontalPositionUnsafe][NumVerticalPositionUnsafe][NumHitByBullet];  
		int count = 0;  
		for (int a = 0; a < NumHeading; a++)  
		  for (int b = 0; b < NumTargetDistance; b++)  
		    for (int c = 0; c < NumTargetBearing; c++)  
		      for (int d = 0; d < NumHorizontalPositionUnsafe; d++)  
		    	for (int e = 0; e < NumVerticalPositionUnsafe; e++)  
		          for (int f = 0; f < NumHitByBullet; f++)  
		      Mapping[a][b][c][d][e][f] = count++;  
		  
		NumStates = count;  
	}
	
	public static int getHeading(double heading)  {  
		double unit = 360 / NumHeading;  
		double newHeading = heading + unit / 2;  
		if (newHeading > 360.0)  
		  newHeading -= 360.0;  
		return (int)(newHeading / unit);  
	} 
	
	public static int getTargetDistance(double value)  {  
	    int distance = (int)(value / 30.0);  
	    if (distance > NumTargetDistance - 1)  
	      distance = NumTargetDistance - 1;  
	    return distance;  
    }
	
	public static int getTargetBearing(double bearing)  {  
		double pi_2 = Math.PI * 2;  
		if (bearing < 0)  
			bearing = pi_2 + bearing;  
		double unit = pi_2 / NumTargetBearing;  
		double newBearing = bearing + unit / 2;  
		if (newBearing > pi_2)  
			newBearing -= pi_2;  
		return (int)(newBearing / unit);  
	} 
	
	public static int getHorizontalPositionUnsafe (double robotX, double BattleFieldX)  {
		int distanceToCenterH;
		if (robotX > 50 || robotX < BattleFieldX-50 ) {
			distanceToCenterH = 0;	// Safe
		} else {
			distanceToCenterH = 1;	// unSafe, too close to wall
		}
		return distanceToCenterH;
	}
	
	public static int getVerticalPositionUnsafe (double robotY, double BattleFieldY)  {
		int distanceToCenterV;
		if (robotY > 50 || robotY < BattleFieldY-50 ) {
			distanceToCenterV = 0;	// Safe
		} else {
			distanceToCenterV = 1;	// unSafe, too close to wall
		}
		return distanceToCenterV;
	}
	
	
	public static int getStateIndex(int heading, int distance, int bearing, int horizontalUnsafe,int verticalUnsafe, int hitbybullet) {
		return Mapping[heading][distance][bearing][horizontalUnsafe][verticalUnsafe][hitbybullet];
	}
	
	public static int[] getStateFromIndex(int index)
	 {
		 int heading = index/(NumTargetDistance*NumTargetBearing*NumHorizontalPositionUnsafe*NumVerticalPositionUnsafe*NumHitByBullet);
		 int remain = index % (NumTargetDistance*NumTargetBearing*NumHorizontalPositionUnsafe*NumVerticalPositionUnsafe*NumHitByBullet);
		 int targetDistances = remain/(NumTargetBearing*NumHorizontalPositionUnsafe*NumVerticalPositionUnsafe*NumHitByBullet);
		 remain = remain % (NumTargetBearing*NumHorizontalPositionUnsafe*NumVerticalPositionUnsafe*NumHitByBullet);
		 int targetBearing = remain/(NumHorizontalPositionUnsafe*NumVerticalPositionUnsafe*NumHitByBullet);
		 remain = remain % (NumHorizontalPositionUnsafe*NumVerticalPositionUnsafe*NumHitByBullet);
		 int horizontalUnsafe = remain/(NumVerticalPositionUnsafe*NumHitByBullet);
		 remain = remain % (NumVerticalPositionUnsafe*NumHitByBullet);
		 int verticalUnsafe = remain/(NumHitByBullet);
		 int hitByBullet = remain % (NumHitByBullet);		 
		 int[] states = new int[6];		 
		 states[0]=heading;
		 states[1]=targetDistances;
		 states[2]=targetBearing;
		 states[3]=horizontalUnsafe;
		 states[4]=verticalUnsafe;
		 states[5]=hitByBullet;
		 
		 return states;
	 }
}
