// javac randomgraph.java
// java randomgraph
// need to modify the matrix display { { // each row , } }
// need to modify the generator instead of 1 or 0, it should attach the the weight for each edge

// Author:	Renee L. Ramsey, William F. Klostermeyer
// Algorithms: An Undergraduate Course with Programming
// Program:     Graph Class
// Date:	2/23/2002
//
// randomgraph.java
// Graph class which has data and methods to work with a graph.
//      Same functionality as "graph.java"
//      Generates random, unweighted, undirected graphs
// Notes: 	The graph class takes two ints and a long as parameters:
//              first is number of vertices
//              second is edge probability (0-100) indicates the
//                    probability that an edge exists
//              third is the long seed for the random number generator
//                    use -1 to have the system clock used
//              Uses adjacency matrix
// To run this program: > Call the class from another program.
//			  Example:  randomgraph a = new randomgraph(12, 50, -1);
//
import java.io.*;	//for I/O
import java.util.Random;
///////////////////////////////////////////////////////////////////////////
public class randomgraph
{
   public  int current_edge_weight;                     // used in
							//    next_neighbor
   private int[][] M;					// adjacency matrix to represent a graph
   private int n;					// number of cols/rows
   private int nVerts;					// number of vertices
   private int x;					// row pointer
   private int y;                                       // column pointer
   private int[] next;                                  // array to track next neighbor

   public randomgraph(int a, int prob, long theseed) 
   {

      double b;			// declare & initialize variables
      int i, j;

      // more initializations
      n = a;						// initialize n to number of cols/rows
      x = 0; 						// initialize number of rows to zero
      y = 0;                                            // initialize number of columns to zero
      M = new int[n][n];             			// initialize 2D array to all zeros
      nVerts = n;					// initialize number of vertices
      next = new int[n];                  		// next neighbor for each vertex

      for(i=0; i < nVerts; i++)			// initialize next neighbor
           next[i]=-1;

      Random generator = new Random();
      Random generator2 = new Random(theseed);
      Random mygenerator;
      Random r = new Random(); // generate the weight

      if (theseed == -1)
        mygenerator = generator;
      else
        mygenerator = generator2;

      for(i=0; i < nVerts; i++) {
         for(j=0; j < nVerts; j++) {

            if (i == j) 
               M[i][j]=0;
            else if (j < i)
               M[i][j] = M[j][i];
            else {
                  b = mygenerator.nextDouble() * 100;
                  int randomWeight = r.nextInt(100) + 1;
                  if (b <= prob)
                     M[i][j] = randomWeight;
                  else
                     M[i][j] = 0;                  
                 }
         }
      }


   }// end constructor
//------------------------------------------------------------------------
   public void insertVertex(int a, int x, int y)	// insert a vertex
   {
      if(x == y)					// if M[i][i]
      {
         if(a != 0)                                     // if value if not zero, display error and exit
         {
            System.out.println("Cannot initialize graph, M[i][i] must be zero!  Exiting...");
            System.exit(0);
         }// end if
      }// end outer if

      M[x][y] = a;					// insert vertex into matrix M

   }// end method insertVertex()
//------------------------------------------------------------------------
   public void display()
   {
    System.out.println("");    				// display the array
    for(int row=0; row<n; row++)
      {
        System.out.print("{ ");
      	for(int col=0; col<n; col++)
      	    if (col != n-1) {
      	        System.out.print(M[row][col] + ", ");
      	    } else {
      	        System.out.print(M[row][col]);
      	    }

      	  System.out.println("},");
      }// end for
   }// end method display()
//------------------------------------------------------------------------
   public int vertices()
   {
      return nVerts;					// return the number of vertices

   }// end method vertices()
//------------------------------------------------------------------------
   public int edgeLength(int a, int b)
   {
      return M[a][b];					// return the edge length

   }// end method edgeLength()
//------------------------------------------------------------------------
   public int nextneighbor(int v)
   {

      next[v] = next[v] + 1; 				// initialize next[v] to the next neighbor

      if(next[v] < nVerts)
      {
      	while(M[v][next[v]] == 0 && next[v] < nVerts)
      	{
         next[v] = next[v] + 1;                         // initialize next[v] to the next neighbor

         if(next[v] == nVerts)
         	break;
      	}// end while

     }// end if

      if(next[v] >= nVerts)
      {
         next[v]=-1;                                    // reset to -1
         current_edge_weight = -1;
      }
      else current_edge_weight = M[v][next[v]];

      return next[v];      				// return next neighbor of v to be processed

   }// end method nextneighbor
//---------------------------------------------------------------------------
   public void resetnext()
   {
      for (int i=0; i < nVerts; i++)	// reset the array next to all -1's
         next[i] = -1;

   }// end method resetnext()

   public static void main(String[] args) {
        randomgraph a = new randomgraph(100, 50, -1);
        a.display();
   }

}// end class graph
////////////////////////////////////////////////////////////////////////////


