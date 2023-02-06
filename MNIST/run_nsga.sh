
#variables:

# loop
for i in {1..10}; do
    echo "nsga #$i Running ..."
    python nsga2_method.py $i &   
done

