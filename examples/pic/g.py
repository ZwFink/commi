import pandas as pd
import sys

def count_total_particles(filename):
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(filename)
        
        # Group by Iteration and sum the End Particles
        result = df.groupby('Iteration')['End Particles'].sum().reset_index()
        
        # Print the results
        print("Iteration, Total Particles")
        for _, row in result.iterrows():
            print(f"{int(row['Iteration'])}, {row['End Particles']}")
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_particles.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    count_total_particles(filename)
