#!/bin/sh
echo "Start to make Ctrl update package...."
mkdir -p /home/FUSION/Tmp
rm -rf /home/FUSION/Tmp/ctrl_package /home/FUSION/Tmp/ctrl_package.tar

mkdir -p /home/FUSION/Tmp/ctrl_package/system
echo "#!/bin/sh" >/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
echo "echo \"Updating Ctrl system...\"">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
echo "cd /home/FUSION/Tmp/ctrl_package/system">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
echo "cp -rf bin lib Shell /home/FUSION/">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
cp -f /home/FUSION/install_support/sio_gpio.ko /home/FUSION/bin
chmod 755 /home/FUSION/bin/sio_gpio.ko
#echo "cp -rf Config/cfg_default Config/pvt /home/FUSION/Config/">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
echo "chmod -R 755 /home/FUSION/bin">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
echo "chmod -R 755 /home/FUSION/lib">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
echo "chmod -R 755 /home/FUSION/Shell">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
echo "sh /home/FUSION/Shell/settags.sh">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
echo "chmod -R 644 /home/FUSION/Tags">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
#echo "chmod -R 644 /home/FUSION/Config">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
echo "sync">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
echo "echo \"Update Ctrl system finish\"">>/home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
chmod 755 /home/FUSION/Tmp/ctrl_package/system/FXAutoRun.sh
#cp -rf /home/FUSION/bin /home/FUSION/lib /home/FUSION/Shell /home/FUSION/Config /home/FUSION/Tmp/ctrl_package/system/
cp -rf /home/FUSION/bin /home/FUSION/lib /home/FUSION/Shell /home/FUSION/Tmp/ctrl_package/system/
cd /home/FUSION/Tmp
tar cf ctrl_package.tar ctrl_package
echo "Make Ctrl update package success under /home/FUSION/Tmp/ctrl_package.tar"
