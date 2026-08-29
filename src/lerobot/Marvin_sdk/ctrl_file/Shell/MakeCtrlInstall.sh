#!/bin/sh
echo "Start to make Ctrl install package...."
mkdir -p /home/FUSION/Tmp/install

echo "#!/bin/sh" >/home/FUSION/Tmp/install/FXAutoRun.sh
echo "echo \"Installing Ctrl...\"">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "mount -o remount rw /">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "rm -rf /home/FUSION">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "rm -rf /mnt/FUSION">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "update-rc.d -f FXStart.sh remove">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "mkdir -p /home/FUSION">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "mkdir -p /mnt/FUSION">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "cp -rf bin lib Shell /home/FUSION/">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "cp -rf Config /mnt/FUSION/">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "mkdir -p /mnt/FUSION/log">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "mkdir -p /mnt/FUSION/Tmp">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "cp ./install_support/FXStart.sh /etc/init.d/FXStart.sh">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "cp ./install_support/sio_gpio.ko /home/FUSION/bin/sio_gpio.ko">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "cd /home/FUSION">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "ln -s /mnt/FUSION/Config Config">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "ln -s /mnt/FUSION/log log">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "ln -s /mnt/FUSION/Tmp Tmp">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "chmod -R 755 /home/FUSION/bin">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "chmod -R 755 /home/FUSION/lib">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "chmod -R 755 /home/FUSION/Shell">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "chmod -R 644 /mnt/FUSION/Config">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "chmod -R 644 /mnt/FUSION/log">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "chmod -R 644 /mnt/FUSION/Tmp">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "chmod -R 755 /etc/init.d/FXStart.sh">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "update-rc.d FXStart.sh start 90 2 3 4 5 .">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "sh /home/FUSION/Shell/settags.sh">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "chmod -R 644 /home/FUSION/Tags">>/home/FUSION/Tmp/install/FXAutoRun.sh
# set update version before install
#UPDATE_VERSION='Marvin1.0.0.0'
#echo "echo $UPDATE_VERSION>/home/FUSION/VERSION">>/home/FUSION/Tmp/install/FXAutoRun.sh
#echo "chmod 644 /home/FUSION/VERSION">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "sync">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "mount -o remount ro /">>/home/FUSION/Tmp/install/FXAutoRun.sh
echo "echo \"Install Ctrl finish\"">>/home/FUSION/Tmp/install/FXAutoRun.sh
chmod 755 /home/FUSION/Tmp/install/FXAutoRun.sh
cd /home/FUSION
cp -rf bin lib Shell Config install_support /home/FUSION/Tmp/install/
cd /home/FUSION/Tmp
tar cf ctrl_install_Marvin.tar install
echo "Make Ctrl install package success under /home/FUSION/Tmp/ctrl_install_Marvin.tar"
