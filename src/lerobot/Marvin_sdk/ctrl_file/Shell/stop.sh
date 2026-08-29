#!/bin/sh

kill -2 $(pidof FXTcpFileServer)
kill -2 $(pidof FXPSI)
sleep 1
rmmod FXSI
sleep 1
rmmod FXKMemory

