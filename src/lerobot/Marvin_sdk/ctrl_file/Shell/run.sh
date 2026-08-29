#!/bin/bash

update_dir(){
for file in `ls $1`
do
  if [ -d $1"/"$file ]
  then
  sh $1"/"$file"/"FXAutoRun.sh
  fi
done
}


if [ -f /home/FUSION/Tmp/interfaces ]; then
	filesize_a=$(ls -l /home/FUSION/Tmp/interfaces | awk '{print $5}')
	if [ $filesize_a -gt 1 ]; then
		mount -o remount rw /
		sleep 3
		mv /home/FUSION/Tmp/interfaces /etc/network/interfaces
		sleep 1
		sync
		sleep 1
		mount -o remount ro /
		sudo ifdown -a 
		sleep 5
		sudo ifup -a
		sleep 5
	fi
fi


filesize_b=$(ls -l /etc/network/interfaces | awk '{print $5}')
if [ $filesize_b -lt 10 ]; then
	mount -o remount rw /
	echo "auto lo" >/home/FUSION/Tmp/interfaces	
	echo "iface lo inet loopback" >>/home/FUSION/Tmp/interfaces
	echo "auto br0" >>/home/FUSION/Tmp/interfaces
	echo "iface br0 inet static" >>/home/FUSION/Tmp/interfaces
	echo "address 192.168.1.190" >>/home/FUSION/Tmp/interfaces
	echo "netmask 255.255.255.0" >>/home/FUSION/Tmp/interfaces
	echo "gateway 192.168.1.1" >>/home/FUSION/Tmp/interfaces
	echo "bridge_ports eth1 eth3" >>/home/FUSION/Tmp/interfaces
	echo "bridge_stp off" >>/home/FUSION/Tmp/interfaces
	echo "bridge_waitport 0" >>/home/FUSION/Tmp/interfaces
	echo "bridge_fd 0" >>/home/FUSION/Tmp/interfaces
	echo "dns-nameservers 8.8.8.8" >>/home/FUSION/Tmp/interfaces
	echo " " >>/home/FUSION/Tmp/interfaces

	sleep 3
	mv /home/FUSION/Tmp/interfaces /etc/network/interfaces
	sleep 1
	sync
	sleep 1
	mount -o remount ro /
	sudo ifdown -a 
	sleep 5
	sudo ifup -a
	sleep 5
fi





if [ -f /home/FUSION/Tmp/UpdateFlag ]; then
	mount -o remount rw /
	cd /home/FUSION/Tmp
	rm -rf ctrl_package
	tar xf ctrl_package.tar
	update_dir /home/FUSION/Tmp/ctrl_package
	rm /home/FUSION/Tmp/UpdateFlag
	sync
	mount -o remount ro /
fi	

#sh /home/FUSION/Shell/start.sh
exit

