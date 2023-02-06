
#variables:

# loop
for i in {1..10}; do
    echo "ga #$i Running ..."
    python ga_method.py $i &   
done

