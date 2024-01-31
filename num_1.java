package leedCode;
import java.util.*;



public class num_1 {


	public static int[] twoSum(int[] nums, int target) {
		for (int i = 0; i < nums.length; i++) {
			for (int j = i+1; j < nums.length; j++) {
				if (nums[i]+nums[j] == target) {
					return new int[] {i,j};
				}
			}
			
		}
		throw new IllegalArgumentException("NO solution");
        
    }
	
	public static void main(String[] args) {
		int[] arr = {3,2,4};
		int[] aas = twoSum(arr,6);
		System.out.println(Arrays.toString(aas));
	}
}

