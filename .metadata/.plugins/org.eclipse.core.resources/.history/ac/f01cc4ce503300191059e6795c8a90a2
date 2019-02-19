import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.*; 

class Node {
	private Node leftnode;
	private Node rightnode;

	private String attribute_name;
	private boolean leafNode;
	private String leafValue;
	private int nodeNumber;
	private static int no_of_leaf = 0;
	private static int no_of_node = -1;
	private static int tree_depth = -1;
	private Set<String> attributes;
	
	public Node() {
		super();
	}
	
	public Node(String attribute, Node left_node, Node right_node){
		this.attribute_name = attribute;
		this.leftnode = left_node;
		this.rightnode = right_node;
		no_of_node++;
		this.set_LeafNode(Boolean.FALSE);
	}
	
	public Node(String leaf_value){
		this.leafValue = leaf_value;
		no_of_leaf++;
		this.set_LeafNode(Boolean.TRUE);
	}
	
	public String get_Name() {
		return attribute_name;
	}

	public void set_Name(String name) {
		this.attribute_name = name;
	}
	
	public Set<String> get_Attributes() {
		return attributes;
	}

	public void set_Attributes(Set<String> attributes) {
		this.attributes = attributes;
	}
	
	public Node get_Left() {
		return leftnode;
	}

	public void set_Left(Node left) {
		this.leftnode = left;
	}

	public Node get_Right() {
		return rightnode;
	}

	public void set_Right(Node right) {
		this.rightnode = right;
	}
	
	
	public boolean is_a_LeafNode() {
		return leafNode;
	}

	public void set_LeafNode(boolean leaf_node) {
		this.leafNode = leaf_node;
	}

	public String get_LeafValue() {
		return leafValue;
	}

	public void set_LeafValue(String leaf_value) {
		this.leafValue = leaf_value;
	}
	
	public int get_NodeNumber() {
		return nodeNumber;
	}

	public void set_NodeNumber(int node_number) {
		this.nodeNumber = node_number;
	}
		
	public void printDTree(){
		tree_depth++;
		if(this.attribute_name == null){
			System.out.print(" : " + leafValue);
		}
		else{
			System.out.println();
			for(int i=0; i<tree_depth;i++){
				System.out.print(" | ");
			}
			System.out.print(attribute_name + " = 0");
		}

		if(leftnode != null){
			leftnode.printDTree();
			if(this.attribute_name == null){
				System.out.print(" : " + leafValue);
			}
			else{
				System.out.println();
				for(int i=0; i<tree_depth;i++){
					System.out.print(" | ");
				}
				System.out.print(attribute_name + " = 1" );
			}
			rightnode.printDTree();
		}
		tree_depth--;
	}	

public static int get_no_of_node()
{
	return no_of_node;
}
public static int get_no_of_leaf()
{
	return no_of_leaf;
}
}
class Tree {

	private int no_NonLeafNodes = 0;

	public Node consutructDTree(ArrayList<ArrayList<String>> sampleSet, ArrayList<String> attrList) throws FileNotFoundException{
		int neg_count = 0;
		int pos_count = 0;

		for(int i=1; i < sampleSet.size();i++){
			if(sampleSet.get(i).get(sampleSet.get(i).size()-1).equalsIgnoreCase("1")){
				pos_count++;
			}
			else{
				neg_count++;
			}
		}
		if (attrList.isEmpty() || neg_count == sampleSet.size()-1){
			return new Node("0");

		}
		else if(attrList.isEmpty() || pos_count == sampleSet.size()-1){
			return new Node("1");
		}
		else{

			InfoGain calculate_Gain = new InfoGain();
			String bestAttribute = calculate_Gain.bestAttribute(sampleSet,attrList);

			ArrayList<String> attr = new ArrayList<String>();

			HashMap<String, ArrayList<ArrayList<String>>> newMap = InfoGain.mapOnBestAttr(sampleSet, bestAttribute);
			for(String att: attrList){
				if(!att.equalsIgnoreCase(bestAttribute)){
					attr.add(att);
				}
			}


			if (newMap.size() < 2){
				String value = "0";
				if(pos_count > neg_count){
					value = "1";
				}

				return new Node(value);
			}


			return new Node(bestAttribute, consutructDTree(newMap.get("0"),attr),consutructDTree(newMap.get("1"),attr));
		}
	}
	
	public void Treecopy(Node first, Node second){
		second.set_LeafNode(first.is_a_LeafNode());
		second.set_Name(first.get_Name());
		second.set_LeafValue(first.get_LeafValue());

		if(!first.is_a_LeafNode()){
			second.set_Left(new Node());
			second.set_Right(new Node());

		Treecopy(first.get_Left(), second.get_Left());
	   Treecopy(first.get_Right(), second.get_Right());

		}
	}
	
	public List<Node> getListOfLeafNode(Node root){
		List<Node> leafNodeList = new ArrayList<>();
		if(root.is_a_LeafNode()){ 
			leafNodeList.add(root);
		}
		else{
			if(!root.get_Left().is_a_LeafNode()){
				getListOfLeafNode(root.get_Left());
			}
			if(!root.get_Right().is_a_LeafNode()){
				getListOfLeafNode(root.get_Right());
			}
		}
		return leafNodeList;
	}
	
	public void call_NonLeafNodes(Node root){		
		if(!root.is_a_LeafNode()){								
			no_NonLeafNodes++;
			root.set_NodeNumber(no_NonLeafNodes);
			call_NonLeafNodes(root.get_Left());
			call_NonLeafNodes(root.get_Right());
		}
	}
	
	public String majorityClassCal(Node root){
		int negCount = 0;
		int posCount = 0;
		String majority = "0";
		List<Node> leafNodes = getListOfLeafNode(root);
		for(Node node : leafNodes){
			if(node.get_LeafValue().equalsIgnoreCase("1")){
				posCount++;
				
			}
			else{
				negCount++; 
			}
		}
		if(posCount>negCount){
			majority = "1";
		}

		return majority;
	}
	
	public void nodeReplace(Node root, int N){
		if(!root.is_a_LeafNode()){
			if(root.get_NodeNumber() == N){
		
				String leafValueToBeChanged = majorityClassCal(root);
				root.set_LeafNode(Boolean.TRUE);
				root.set_Left(null);
				root.set_Right(null);
				root.set_LeafValue(leafValueToBeChanged);
			}
			else{
				nodeReplace(root.get_Left(), N);
				nodeReplace(root.get_Right(), N);
			}

		}
	}
	
	public int getNoOfNonLeafNodes() {		
		int number = no_NonLeafNodes;
		setNoOfNonLeafNodes(0);
		return number;

	}
	

	public void setNoOfNonLeafNodes(int noNonLeafNodes) {
		this.no_NonLeafNodes = noNonLeafNodes;
	}
	
	public Node pruneDtree(Node root,ArrayList<ArrayList<String>> sampleSet, int k, ArrayList<ArrayList<String>> validationData){
		Node best_tree;
		Node prunning_tree;
		best_tree = new Node();
		Treecopy(root, best_tree);
		
		double bestAccuracyOfTree = tree_Accuracy(best_tree, validationData);
		prunning_tree = new Node();
		for(int i=1; i<=sampleSet.size();i++){
			Treecopy(root, prunning_tree);
			
			Random random = new Random();

			int M = 1 + random.nextInt(k);
			for(int j=0; j<=M; j++){
				call_NonLeafNodes(prunning_tree);			
				int N = getNoOfNonLeafNodes();
				
				if(N>1){
					int P = random.nextInt(N) + 1;
					nodeReplace(prunning_tree, P);
				}
				else{
					break;
				}
			}
			double pruned_accuracy = tree_Accuracy(prunning_tree, validationData);
			if (pruned_accuracy > bestAccuracyOfTree){
				bestAccuracyOfTree = pruned_accuracy;
				Treecopy(prunning_tree, best_tree);
				
			}
		}
		List <Node> leafs= getListOfLeafNode(best_tree);
		int leaf_count=0;
		for(Node n : leafs)
		{
			System.out.println("leafs value : " + n.get_LeafValue()); 
			leaf_count++;
		}
		System.out.println("no.of.leaf nodes in root" + leaf_count);
		
		return best_tree;
	}

	public boolean verifyClass(Node root, ArrayList<String> row, ArrayList<String> attrList){
		Node nodeCopy = root;
		while(true){
			if(nodeCopy.is_a_LeafNode()){
				if(nodeCopy.get_LeafValue().equalsIgnoreCase(row.get(row.size()-1))){
					return true;
				}
				else{
					return false;
				}
			}

			int index = attrList.indexOf(nodeCopy.get_Name());
			String value = row.get(index);
			if(value.equalsIgnoreCase("0")){
				nodeCopy = nodeCopy.get_Left();
			}
			else{
				nodeCopy = nodeCopy.get_Right();
			}
		}
	}

	public double tree_Accuracy(Node node, ArrayList<ArrayList<String>> dataToBeChecked){
		double accuracy = 0;
		int positiveExamples = 0;

		ArrayList<String> attributes = dataToBeChecked.get(0);
		for(ArrayList<String> row : dataToBeChecked.subList(1, dataToBeChecked.size())){	
			boolean exampleCheck = verifyClass(node, row, attributes);					
			if(exampleCheck){
				positiveExamples++;
			}
		}
		accuracy = (((double) positiveExamples / (double) (dataToBeChecked.size()-1)) * 100.00);

		return accuracy;
	}
}


public class id3 {
	
	
	public ArrayList<ArrayList<String>> readData(String fileName) throws IOException{

		ArrayList<ArrayList<String>> data = new ArrayList<ArrayList<String>>();
		File file = new File(fileName);
		Scanner input;
		input = new Scanner(file);
		while(input.hasNext()){
			String[] dataForEachRow = input.next().split(",");
			data.add(new ArrayList<String>(Arrays.asList(dataForEachRow)));

		}
		input.close();
		return data;
	}


	public static void main(String[] args) throws NumberFormatException, IOException {
									
			String trainingDatafile= args[0];
		 	String validationDatafile = args [1]; 
		    String testFile = args [2];
			
			id3 id3algo = new id3();
			Tree tree = new Tree();

			try {

				ArrayList<ArrayList<String>> dataSetTraining = id3algo.readData(trainingDatafile);
				ArrayList<ArrayList<String>> dataSetValidation = id3algo.readData(validationDatafile);
				ArrayList<ArrayList<String>> dataSetTest = id3algo.readData(testFile);

				ArrayList<String> attributeList = dataSetTraining.get(0);

				int training_instances=dataSetTraining.size()-1;
                int training_attributes=dataSetTraining.get(0).size()-1;
		
               int validation_instances=dataSetValidation.size()-1;
               int validation_attributes=dataSetValidation.get(0).size()-1;
		
                int test_instances=dataSetTest.size()-1;
                int test_attributes=dataSetTest.get(0).size()-1;
		  
            Node trainned_tree_model = tree.consutructDTree(dataSetTraining, attributeList);
            Scanner scan= new Scanner( System.in );
          
            int leafnodes_preprune=Node.get_no_of_leaf();
		    int number_nodes_preprune=Node.get_no_of_node();
			int total_nodes_preprune=leafnodes_preprune+number_nodes_preprune;
		    				
					System.out.println("=============================================Tree======================================================");
					System.out.println();
					
					trainned_tree_model.printDTree();
					
					System.out.println();
					System.out.println();
				System.out.println("===========================================================================================");
				System.out.println("===========================PRE PRUNED ACCURACY==============================================");
                
				System.out.println("the nummber of Training instances:"+training_instances);
                System.out.println("the nummber of Training attributes:"+training_attributes);
                System.out.println("the nummber of leaf nodes:"+leafnodes_preprune);
                System.out.println("the total number of nodes:"+total_nodes_preprune);
                System.out.println(" The Accuracy of the model on the training dataset is : "
						+ tree.tree_Accuracy(trainned_tree_model, dataSetTraining));

                
                System.out.println("the nummber of validation instances:"+validation_instances);
                System.out.println("the nummber of validation attributes:"+validation_attributes);
                System.out.println(" The Accuracy of the model on the validation dataset is : "
						+ tree.tree_Accuracy(trainned_tree_model, dataSetValidation));
                
                System.out.println("the nummber of testing instances:"+test_instances);
                System.out.println("the nummber of testing  attributes:"+test_attributes);
              
                System.out.println(" The Accuracy of the model on the testing dataset is : "
						+ tree.tree_Accuracy(trainned_tree_model, dataSetTest));

				System.out.println();
				System.out.println("============================================================================================");
     
				
				Double prune;
				
				System.out.println("Enter the Pruning Factor");
				prune = scan.nextDouble();
			    int prune_nodes=(int) (total_nodes_preprune*prune);
			    if (prune_nodes == 0 || prune_nodes == 1)
			    {
			    	System.out.println("Re-enter a valid pruning factor excluding 0 and 1 "); 
			    	prune = scan.nextDouble();
				    prune_nodes=(int) (total_nodes_preprune*prune);
			    }
				
				
				System.out.println("nodes to be pruned are :"+prune_nodes);
				System.out.println("total nodes present are :"+total_nodes_preprune);
				
				Node pruned_tree_model = tree.pruneDtree(trainned_tree_model, dataSetTraining, prune_nodes, dataSetValidation);
				
	      
				

				System.out.println("============================================Pruned Tree=======================================");

					System.out.println();
					pruned_tree_model.printDTree();
					System.out.println();
					System.out.println();
					
					
						System.out.println("===========================POST PRUNED ACCURACY==============================================");
		                
						System.out.println("the nummber of Training instances:"+training_instances);
		                System.out.println("the nummber of Training attributes:"+training_attributes);
		                System.out.println("the nummber of leaf nodes"+(total_nodes_preprune-prune_nodes));
		                System.out.println("the total number of nodes" + (total_nodes_preprune-prune_nodes));
		                System.out.println(" The Accuracy of Tree on training set is : "
								+ tree.tree_Accuracy(pruned_tree_model, dataSetTraining));

		                
		                System.out.println("the nummber of validation instances:"+validation_instances);
		                System.out.println("the nummber of validation attributes:"+validation_attributes);
		                System.out.println(" The Accuracy of Tree on validation set is : "
								+ tree.tree_Accuracy(pruned_tree_model, dataSetValidation));
		                
		                System.out.println("the nummber of testing instances:"+test_instances);
		                System.out.println("the nummber of testing  attributes:"+test_attributes);
		              
		                System.out.println(" The Accuracy of Tree on testing set is : "
								+ tree.tree_Accuracy(pruned_tree_model, dataSetTest));

						System.out.println();
						System.out.println("============================================================================================");



			} catch (IOException e) {
				System.out.println("An input file could not be found, try again");

			}
		

	}



		
}


	
class InfoGain {

	HashMap<String, ArrayList<String>> dataMap;
	HashMap<String, Double> gainMap ;

	public static double calculateEntropy(double positive, double negative){
		double total = positive + negative;
		double positiveProbability = positive/total;
		double negativeProbability = negative/total;

		if(positive == negative){
			return 1;
		}
		else if(positive == 0 || negative == 0){
			return 0;
		}
		else{
			double entropy_value = ((-positiveProbability) * (Math.log(positiveProbability)/Math.log(2))) + ((-negativeProbability)*(Math.log(negativeProbability)/Math.log(2)));
			return entropy_value;
		}

	}
	
		
	public double calculateInfoGain(double rootPositive, double rootNegative, double positiveLeft, double negativeLeft, double positiveRight, double negativeRight){
		double totalroot_value = rootPositive + rootNegative;
		double rootEntropy = calculateEntropy(rootPositive, rootNegative);
		double left_entropy = calculateEntropy(positiveLeft,negativeLeft);
		double right_entropy = calculateEntropy(positiveRight, negativeRight);
		double left_total = positiveLeft + negativeLeft;
		double right_total = positiveRight + negativeRight;

		double gain = rootEntropy - (((left_total/totalroot_value)* left_entropy) + ((right_total/totalroot_value) * right_entropy));

		return gain;
	}
	
	public static HashMap<String, ArrayList<String>> populateMap(ArrayList<ArrayList<String>> data) throws FileNotFoundException{
		HashMap<String, ArrayList<String>> map = new HashMap<String, ArrayList<String>>();

		ArrayList<String>flag_key = data.get(0);	

		for(int i=0;i<flag_key.size();i++){
			for(int j=1;j<data.size();j++){
				if (map.containsKey(flag_key.get(i))){
					map.get(flag_key.get(i)).add(data.get(j).get(i));
				}
				else{
					ArrayList<String> values = new ArrayList<String>();
					values.add(data.get(j).get(i));
					map.put(flag_key.get(i), values);
				}
			}
		}
		return map;
	}
	
	public static HashMap<String,ArrayList<ArrayList<String>>> mapOnBestAttr(ArrayList<ArrayList<String>> data, String bestAttr){
		HashMap<String, ArrayList<ArrayList<String>>> reducedMap = new HashMap<String, ArrayList<ArrayList<String>>>();
		int index = data.get(0).indexOf(bestAttr);
		
		for(int i=1;i<data.size();i++){
			if(data.get(i).get(index).equalsIgnoreCase("0")){
				if(reducedMap.containsKey("0")){
					reducedMap.get("0").add(data.get(i));
				}
				else{
					ArrayList<ArrayList<String>> dataAdd = new ArrayList<ArrayList<String>>();
					dataAdd.add(data.get(0));
					dataAdd.add(data.get(i));
					reducedMap.put("0",dataAdd);
				}

			}
			else{
				if(reducedMap.containsKey("1")){
					reducedMap.get("1").add(data.get(i));
				}
				else{
					ArrayList<ArrayList<String>> dataAdd = new ArrayList<ArrayList<String>>();
					dataAdd.add(data.get(0));
					dataAdd.add(data.get(i));
					reducedMap.put("1",dataAdd);
				}
			}
		}

		return reducedMap;
	}


	public String bestAttribute(ArrayList<ArrayList<String>> data, ArrayList<String> attributeList) throws FileNotFoundException{
		String bestAttr = "";
		dataMap = populateMap(data);
		gainMap = new HashMap<String, Double>();
		
		double classPositive = 0;
		double classNegative = 0;
		for(String value : dataMap.get("Class")){
			if(value.equalsIgnoreCase("1")){
				classPositive++;
			}
			else{
				classNegative++;
			}
		}

		for(String key: attributeList.subList(0, attributeList.size()-1)){		
			ArrayList<String> temp = dataMap.get(key);
			double positive_left = 0;
			double positive_right = 0;
			double negative_left = 0;
			double negative_right = 0;
			for(int i=0; i<temp.size();i++){								
				if(temp.get(i).equalsIgnoreCase("0")){
					if(dataMap.get("Class").get(i).equalsIgnoreCase("1")){
						positive_left++;
					}
					else{
						negative_left++;
					}
				}
				else{
					if(dataMap.get("Class").get(i).equalsIgnoreCase("1")){
						positive_right++;
					}
					else{
						negative_right++;
					}
				}
			}

	
				Double gainForEachKey = calculateInfoGain(classPositive, classNegative, positive_left, negative_left, positive_right, negative_right);
				gainMap.put(key, gainForEachKey);
				
		
		}

		ArrayList<Double> values_list = new ArrayList<Double>(gainMap.values());
		Collections.sort(values_list);
		Collections.reverse(values_list);
		for(String key: gainMap.keySet()){
			if (values_list.get(0).equals(gainMap.get(key))){
				bestAttr = key;
				break;
			}
		}
		return bestAttr;		
	}

}
