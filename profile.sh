echo "Running profiling..."

threads=(10 8 4 2 1)

width=1500
height=800
samples=100

echo -e "width: $width \theight: $height \tsamples: $samples \tthreads: ${threads[*]}"

prefix="Render time:"

function cancel {
	echo "Received SIGQUIT. Exiting."
	exit 0;
}

trap cancel SIGQUIT

for scene in `seq 0 2`
do
	echo -n "GPU BVH $scene..."
	file="GPU_BVH"$scene".out"
	./PathTracer -w $width -h $height -p $scene -d 0 -a 1>$file
	timing="$(grep "$prefix" $file)"
	echo "$timing" | sed -e "s/^$prefix//"
done

for scene in `seq 0 2`
do
	echo -n "GPU scene $scene..."
	file="GPU_scene"$scene".out"
	./PathTracer -w $width -h $height -s $samples -p $scene -d 0 1>$file
	timing="$(grep "$prefix" $file)"
	echo "$timing" | sed -e "s/^$prefix//"
done

for scene in `seq 0 2`
do
	for i in ${threads[@]}
	do
		echo -n "CPU BVH $scene threads $i..."
		file="CPU_BVH_"$scene"_threads_"$i".out"
       	./PathTracer -w $width -h $height -p $scene -d 1 -t $i -a 1>$file
		timing="$(grep "$prefix" $file)"
		echo "$timing" | sed -e "s/^$prefix//"
	done
done

for scene in `seq 0 2`
do
	for i in ${threads[@]}
	do
		echo -n "CPU scene $scene threads $i..."
		file="CPU_scene_"$scene"_threads_"$i".out"
       	./PathTracer -w $width -h $height -s $samples -p $scene -d 1 -t $i 1>$file
		timing="$(grep "$prefix" $file)"
		echo "$timing" | sed -e "s/^$prefix//"
	done
done
