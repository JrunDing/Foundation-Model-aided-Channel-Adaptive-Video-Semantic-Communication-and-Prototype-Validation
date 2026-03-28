"""
@Brief: MIMO OFDM for video transmission FVSC.
"""

import numpy as np
import commpy as cpy
import warnings
import matplotlib.pyplot as plt
from channel import Channel
from commpy.channelcoding.ldpc import get_ldpc_code_params, ldpc_bp_decode, triang_ldpc_systematic_encode


warnings.filterwarnings("ignore")


class MIMO_OFDM(object):
    """
    :brief: MIMO-OFDM communication system. Slow fading channel.
    """
    def __init__(self, noTX, noRX, K, CE, SD, moduType='QAM16', channel_model="UMi", pilotValue=3+3j):
        self.noTX = noTX
        self.noRX = noRX

        self.moduType = moduType  
        self.m_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "QAM16": 4, "QAM64": 6}
        self.mu = self.m_map[self.moduType]  

        self.CE = CE
        self.SD = SD

        self.K = K 
        self.CP = self.K // 4  
        self.pilotValue = pilotValue  

        self.h11Carriers = np.arange(0, 64, 2)  
        self.h21Carriers = np.arange(0, 64, 2) 
        self.h12Carriers = np.arange(1, 64, 2)  
        self.h22Carriers = np.arange(1, 64, 2)  
        self.allCarriers = np.arange(0, 64)  

        self.chanenl_object = Channel()
        if channel_model == "UMi":
            self.H_freq = self.chanenl_object.gen_ch_2_2_UMi_NLOS()  
        elif channel_model == "UMa":
            self.H_freq = self.chanenl_object.gen_ch_2_2_UMa_NLOS()  
        elif channel_model == "RMa":
            self.H_freq = self.chanenl_object.gen_ch_2_2_RMa_NLOS()  
        elif channel_model == "Indoor":
            self.H_freq = self.chanenl_object.gen_ch_2_2_Indoor() 
        elif channel_model == "OTA":
            pass

    def get_H_f(self, channel_model, idx):
        if channel_model == "OTA":
            self.H_freq = self.chanenl_object.gen_ch_2_2_OTA(idx)
        else:
            pass
        return self.H_freq

    def modulation(self, bits):
        if self.moduType == "QPSK":
            PSK4 = cpy.PSKModem(4)
            symbol = PSK4.modulate(bits)
            return symbol
        elif self.moduType == "QAM64":
            QAM64 = cpy.QAMModem(64)
            symbol = QAM64.modulate(bits)
            return symbol
        elif self.moduType == "QAM16":
            QAM16 = cpy.QAMModem(16)
            symbol = QAM16.modulate(bits)
            return symbol
        elif self.moduType == "8PSK":
            PSK8 = cpy.PSKModem(8)
            symbol = PSK8.modulate(bits)
            return symbol
        elif self.moduType == "BPSK":
            BPSK = cpy.PSKModem(2)
            symbol = BPSK.modulate(bits)
            return symbol

    def demodulation(self, symbol):
        if self.moduType == "QPSK":
            PSK4 = cpy.PSKModem(4)
            bits = PSK4.demodulate(symbol, demod_type='hard')
            return bits
        elif self.moduType == "QAM64":
            QAM64 = cpy.QAMModem(64)
            bits = QAM64.demodulate(symbol, demod_type='hard')
            return bits
        elif self.moduType == "QAM16":
            QAM16 = cpy.QAMModem(16)
            bits = QAM16.demodulate(symbol, demod_type='hard')
            return bits
        elif self.moduType == "8PSK":
            PSK8 = cpy.PSKModem(8)
            bits = PSK8.demodulate(symbol, demod_type='hard')
            return bits
        elif self.moduType == "BPSK":
            BPSK = cpy.PSKModem(2)
            bits = BPSK.demodulate(symbol, demod_type='hard')
            return bits

    def layerMap(self, symbols):
        dataFlow1 = symbols[:int(len(symbols)/2)]
        dataFlow2 = symbols[int(len(symbols)/2):]
        return dataFlow1, dataFlow2

    def layerMerge(self, dataFlow):
        symbols = np.concatenate((dataFlow[:, 0], dataFlow[:, 1]))
        return symbols

    def channel(self, txOFDM, H_freq):
        rxOFDM = np.zeros((self.K, self.noRX), dtype=np.complex128)  
        averAmp = np.sqrt(np.mean(np.square(np.abs(txOFDM)))) 
        txOFDM = txOFDM / averAmp 
        for i in range(self.K):
            rxOFDM[i, :] = H_freq[i, :, :] @ txOFDM[i, :] 
        return rxOFDM, averAmp

    def addNoise(self, rxOFDM, SNRdB):
        sigPwr = np.mean(np.square(np.abs(rxOFDM)))
        noisePwr = sigPwr / (10 ** (SNRdB / 10))
        noise = 1 / np.sqrt(2) * (np.random.randn(self.K, self.noRX) + 1j * np.random.randn(self.K, self.noRX)) * np.sqrt(noisePwr)
        rxOFDM = rxOFDM + noise
        return rxOFDM

    def channelEstimation(self, algorithm, rxBlockPilotOFDM, rxBlockPilotOFDM_2, averAmp, averAmp_2):
        if algorithm == "LS":
            pilotValueTemp = self.pilotValue / averAmp 
            H11 = (rxBlockPilotOFDM[:, 0][self.h11Carriers] / pilotValueTemp)  
            H21 = (rxBlockPilotOFDM[:, 1][self.h21Carriers] / pilotValueTemp)  
            H12 = (rxBlockPilotOFDM[:, 0][self.h12Carriers] / pilotValueTemp) 
            H22 = (rxBlockPilotOFDM[:, 1][self.h22Carriers] / pilotValueTemp)  

            pilotValueTemp_2 = self.pilotValue / averAmp_2 
            H11_2 = (rxBlockPilotOFDM_2[:, 0][np.arange(1, 64, 2)] / pilotValueTemp_2)  
            H21_2 = (rxBlockPilotOFDM_2[:, 1][np.arange(1, 64, 2)] / pilotValueTemp_2)  
            H12_2 = (rxBlockPilotOFDM_2[:, 0][np.arange(0, 64, 2)] / pilotValueTemp_2)  
            H22_2 = (rxBlockPilotOFDM_2[:, 1][np.arange(0, 64, 2)] / pilotValueTemp_2) 

            H11Est = np.empty((H11.size + H11_2.size), dtype=H11.dtype)
            H11Est[0::2] = H11
            H11Est[1::2] = H11_2

            H21Est = np.empty((H21.size + H21_2.size), dtype=H21.dtype)
            H21Est[0::2] = H21
            H21Est[1::2] = H21_2

            H12Est = np.empty((H12.size + H12_2.size), dtype=H12.dtype)
            H12Est[0::2] = H12_2
            H12Est[1::2] = H12

            H22Est = np.empty((H22.size + H22_2.size), dtype=H22.dtype)
            H22Est[0::2] = H22_2
            H22Est[1::2] = H22
            return H11Est, H21Est, H12Est, H22Est  

    def genWZf(self, H11Est, H21Est, H12Est, H22Est):
        H = np.stack((np.stack((H11Est, H12Est), axis=1), np.stack((H21Est, H22Est), axis=1)), axis=1)
        w_ZF = np.zeros((self.K, self.noRX, self.noTX), dtype=np.complex128)
        for i in range(self.K): w_ZF[i, :, :] = np.linalg.inv(H[i, :, :].conj().T @ H[i, :, :]) @ H[i, :, :].conj().T 
        return w_ZF

    def genWMMSE(self, H11Est, H21Est, H12Est, H22Est, noisePwr):
        H = np.stack((np.stack((H11Est, H12Est), axis=1), np.stack((H21Est, H22Est), axis=1)), axis=1)
        wMMSE = np.zeros((self.K, self.noRX, self.noTX), dtype=np.complex128)

        for j in range(self.K): wMMSE[j, :, :] = (np.linalg.inv(H[j, :, :].conj().T @ H[j, :, :] +
                                                                noisePwr * np.eye(self.noRX, dtype=np.complex128))) \
                                                 @ H[j, :, :].conj().T  
        return wMMSE

    def signalDetectionZf(self, rxFlow, wZf):
        rxFlow_ = np.zeros((self.K, self.noRX), dtype=np.complex128)
        for i in range(self.K): rxFlow_[i, :] = wZf[i, :, :] @ rxFlow[i, :]
        return rxFlow_

    def signalDetectionMMSE(self, rxFlow, wMMSE):
        rxFlow_ = np.zeros((self.K, self.noRX), dtype=np.complex128)
        for i in range(self.K): rxFlow_[i, :] = wMMSE[i, :, :] @ rxFlow[i, :]
        return rxFlow_

    def MIMO_OFDM_Simulation(self, txBits, SNRdB):
        global rxFlowDetected, W
        txBlockPilotOFDM1 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM1[np.arange(0, self.K, 2), :] = self.pilotValue  # TX1 OFDM
        txBlockPilotOFDM2 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM2[np.arange(1, self.K, 2), :] = self.pilotValue  # TX2 OFDM
        txBlockPilotOFDM = np.concatenate((txBlockPilotOFDM1, txBlockPilotOFDM2), axis=1)  

        rxBlockPilotOFDM, averAmp = self.channel(txBlockPilotOFDM, self.H_freq)  
        rxBlockPilotOFDM = self.addNoise(rxBlockPilotOFDM, SNRdB)  

        txBlockPilotOFDM1_2 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM1_2[np.arange(1, self.K, 2), :] = self.pilotValue  
        txBlockPilotOFDM2_2 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM2_2[np.arange(0, self.K, 2), :] = self.pilotValue  
        txBlockPilotOFDM_2 = np.concatenate((txBlockPilotOFDM1_2, txBlockPilotOFDM2_2), axis=1)  

        rxBlockPilotOFDM_2, averAmp_2 = self.channel(txBlockPilotOFDM_2, self.H_freq)  
        rxBlockPilotOFDM_2 = self.addNoise(rxBlockPilotOFDM_2, SNRdB) 

        if self.CE == "Perfect":
            H11Est = self.H_freq[:, 0, 0]
            H21Est = self.H_freq[:, 1, 0]
            H12Est = self.H_freq[:, 0, 1]
            H22Est = self.H_freq[:, 1, 1]
        else:
            H11Est, H21Est, H12Est, H22Est = self.channelEstimation(self.CE, rxBlockPilotOFDM, rxBlockPilotOFDM_2, averAmp, averAmp_2)

        noBitsOnce = self.mu*self.K*self.noTX  
        if len(txBits) % noBitsOnce:
            supNoBits = noBitsOnce - (len(txBits) % noBitsOnce)  
            txBits = np.concatenate((txBits, np.zeros(supNoBits, dtype=np.int32))) 
        else:
            supNoBits = 0
        txSymbols = self.modulation(txBits)
        noTransmit = int(len(txSymbols) / (self.K*self.noTX)) 
        rxSymbols = np.array([], dtype=np.complex128)
        for i in range(noTransmit):
            txSymbols_ = txSymbols[i*(self.K*self.noTX):(i+1)*(self.K*self.noTX)]
            txFlow1, txFlow2 = self.layerMap(txSymbols_)
            txFlow = np.stack((txFlow1, txFlow2), axis=1)  
            rxFlow, averAmp = self.channel(txFlow, self.H_freq)  
            rxFlow_ = self.addNoise(rxFlow, SNRdB)  
            if self.SD == "ZF":
                W = self.genWZf(H11Est, H21Est, H12Est, H22Est)
                rxFlowDetected = self.signalDetectionZf(rxFlow_, W) 
            elif self.SD == "MMSE":
                noisePwr = (np.mean(np.square(np.abs(rxFlow)))) / (10 ** (SNRdB / 10))
                W = self.genWMMSE(H11Est, H21Est, H12Est, H22Est, noisePwr)
                rxFlowDetected = self.signalDetectionMMSE(rxFlow_, W)

            rxFlowDetected = rxFlowDetected*averAmp 
            rxSymbols_ = self.layerMerge(rxFlowDetected)
            rxSymbols = np.concatenate((rxSymbols, rxSymbols_))
        if supNoBits == 0: rxBits = self.demodulation(rxSymbols)
        else: rxBits = self.demodulation(rxSymbols)[:-supNoBits]
        return rxBits

    def MIMO_OFDM_Simulation_Modulation(self, txBits, SNRdB, modulation_method):
        """
        :brief: MIMO-OFDM simulation  one slot/frame transmission，i.e., one pilot OFDM and multiple data OFDM
                transmission process :: ::::::   : represents an OFDM symbol，the first two are used for channel estimation, and the latter is used for data transmission
        :param txBits: tx bits  one-dimensional numpy
        :param SNRdB: signal-to-noise ratio (SNR) dB
        :param modulation_method: modulation type
        :return: rx bits  one-dimensional numpy
        """
        if modulation_method == "QPSK":
            modulator = cpy.PSKModem(4)
        elif modulation_method == "QAM64":
            modulator = cpy.QAMModem(64)
        elif modulation_method == "QAM16":
            modulator = cpy.QAMModem(16)
        elif modulation_method == "8PSK":
            modulator = cpy.PSKModem(8)
        else:  # BPSK
            modulator = cpy.PSKModem(2)
        mu = self.m_map[modulation_method]  # bit number of one symbol

        global rxFlowDetected, W
        ############################################################
        # 1. transmit channel estimation OFDM symbols
        # transmit the first group of grid-based OFDM symbols for channel estimation
        txBlockPilotOFDM1 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM1[np.arange(0, self.K, 2), :] = self.pilotValue  # TX1 OFDM
        txBlockPilotOFDM2 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM2[np.arange(1, self.K, 2), :] = self.pilotValue  # TX2 OFDM
        txBlockPilotOFDM = np.concatenate((txBlockPilotOFDM1, txBlockPilotOFDM2), axis=1)  # TX OFDM
        # wireless channel
        rxBlockPilotOFDM, averAmp = self.channel(txBlockPilotOFDM, self.H_freq)
        rxBlockPilotOFDM = self.addNoise(rxBlockPilotOFDM, SNRdB)

        # transmit the second group of grid-based OFDM symbols for channel estimation
        txBlockPilotOFDM1_2 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM1_2[np.arange(1, self.K, 2), :] = self.pilotValue  # TX1 OFDM
        txBlockPilotOFDM2_2 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM2_2[np.arange(0, self.K, 2), :] = self.pilotValue  # TX2 OFDM
        txBlockPilotOFDM_2 = np.concatenate((txBlockPilotOFDM1_2, txBlockPilotOFDM2_2), axis=1)  # TX OFDM
        # wireless channel
        rxBlockPilotOFDM_2, averAmp_2 = self.channel(txBlockPilotOFDM_2, self.H_freq)
        rxBlockPilotOFDM_2 = self.addNoise(rxBlockPilotOFDM_2, SNRdB)

        # channel estimation
        if self.CE == "Perfect":
            # assume perfect channel estimation
            H11Est = self.H_freq[:, 0, 0]
            H21Est = self.H_freq[:, 1, 0]
            H12Est = self.H_freq[:, 0, 1]
            H22Est = self.H_freq[:, 1, 1]
        else:
            # practical channel estimation
            H11Est, H21Est, H12Est, H22Est = self.channelEstimation(self.CE, rxBlockPilotOFDM, rxBlockPilotOFDM_2, averAmp, averAmp_2)

        ############################################################
        # 2. transmit data OFDM symbols
        noBitsOnce = mu*self.K*self.noTX  # number of bits transmitted per transmission   mu×64×2=m×128
        if len(txBits) % noBitsOnce:
            supNoBits = noBitsOnce - (len(txBits) % noBitsOnce)
            txBits = np.concatenate((txBits, np.zeros(supNoBits, dtype=np.int32)))  # speed matching
        else:
            supNoBits = 0
        txSymbols = modulator.modulate(txBits)
        noTransmit = int(len(txSymbols) / (self.K*self.noTX))  # number of transmissions
        rxSymbols = np.array([], dtype=np.complex128)
        # transmission
        for i in range(noTransmit):
            txSymbols_ = txSymbols[i*(self.K*self.noTX):(i+1)*(self.K*self.noTX)]
            txFlow1, txFlow2 = self.layerMap(txSymbols_)
            txFlow = np.stack((txFlow1, txFlow2), axis=1)  # 64×2
            # wireless channel
            rxFlow, averAmp = self.channel(txFlow, self.H_freq)
            rxFlow_ = self.addNoise(rxFlow, SNRdB)
            # signal detection
            if self.SD == "ZF":
                W = self.genWZf(H11Est, H21Est, H12Est, H22Est)
                rxFlowDetected = self.signalDetectionZf(rxFlow_, W)  # 64×2
            elif self.SD == "MMSE":
                noisePwr = (np.mean(np.square(np.abs(rxFlow)))) / (10 ** (SNRdB / 10))
                W = self.genWMMSE(H11Est, H21Est, H12Est, H22Est, noisePwr)
                rxFlowDetected = self.signalDetectionMMSE(rxFlow_, W)

            rxFlowDetected = rxFlowDetected*averAmp  # amplify the signal
            rxSymbols_ = self.layerMerge(rxFlowDetected)
            rxSymbols = np.concatenate((rxSymbols, rxSymbols_))
        if supNoBits == 0: rxBits = modulator.demodulate(rxSymbols, demod_type='hard')
        else: rxBits = modulator.demodulate(rxSymbols, demod_type='hard')[:-supNoBits]
        return rxBits

    def MIMO_OFDM_Simulation_OTA(self, txBits, cha_idx, SNRdB):
        self.H_freq = self.chanenl_object.gen_ch_2_2_OTA(cha_idx)
        global rxFlowDetected, W
        txBlockPilotOFDM1 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM1[np.arange(0, self.K, 2), :] = self.pilotValue  
        txBlockPilotOFDM2 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM2[np.arange(1, self.K, 2), :] = self.pilotValue 
        txBlockPilotOFDM = np.concatenate((txBlockPilotOFDM1, txBlockPilotOFDM2), axis=1)  

        rxBlockPilotOFDM, averAmp = self.channel(txBlockPilotOFDM,
                                                 self.H_freq) 
        rxBlockPilotOFDM = self.addNoise(rxBlockPilotOFDM, SNRdB)  

        txBlockPilotOFDM1_2 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM1_2[np.arange(1, self.K, 2), :] = self.pilotValue 
        txBlockPilotOFDM2_2 = np.zeros((self.K, 1), dtype=np.complex128)
        txBlockPilotOFDM2_2[np.arange(0, self.K, 2), :] = self.pilotValue  
        txBlockPilotOFDM_2 = np.concatenate((txBlockPilotOFDM1_2, txBlockPilotOFDM2_2), axis=1)  

        rxBlockPilotOFDM_2, averAmp_2 = self.channel(txBlockPilotOFDM_2, self.H_freq) 
        rxBlockPilotOFDM_2 = self.addNoise(rxBlockPilotOFDM_2, SNRdB)  

        if self.CE == "Perfect":
            H11Est = self.H_freq[:, 0, 0]
            H21Est = self.H_freq[:, 1, 0]
            H12Est = self.H_freq[:, 0, 1]
            H22Est = self.H_freq[:, 1, 1]
        else:
            H11Est, H21Est, H12Est, H22Est = self.channelEstimation(self.CE, rxBlockPilotOFDM, rxBlockPilotOFDM_2,
                                                                    averAmp, averAmp_2)

        noBitsOnce = self.mu * self.K * self.noTX  
        if len(txBits) % noBitsOnce:
            supNoBits = noBitsOnce - (len(txBits) % noBitsOnce)  
            txBits = np.concatenate((txBits, np.zeros(supNoBits, dtype=np.int32)))  
        else:
            supNoBits = 0
        txSymbols = self.modulation(txBits)
        noTransmit = int(len(txSymbols) / (self.K * self.noTX)) 
        rxSymbols = np.array([], dtype=np.complex128)

        for i in range(noTransmit):
            txSymbols_ = txSymbols[i * (self.K * self.noTX):(i + 1) * (self.K * self.noTX)]
            txFlow1, txFlow2 = self.layerMap(txSymbols_)
            txFlow = np.stack((txFlow1, txFlow2), axis=1) 
            rxFlow, averAmp = self.channel(txFlow, self.H_freq)  
            rxFlow_ = self.addNoise(rxFlow, SNRdB) 
            if self.SD == "ZF":
                W = self.genWZf(H11Est, H21Est, H12Est, H22Est)
                rxFlowDetected = self.signalDetectionZf(rxFlow_, W)  
            elif self.SD == "MMSE":
                noisePwr = (np.mean(np.square(np.abs(rxFlow)))) / (10 ** (SNRdB / 10))
                W = self.genWMMSE(H11Est, H21Est, H12Est, H22Est, noisePwr)
                rxFlowDetected = self.signalDetectionMMSE(rxFlow_, W)

            rxFlowDetected = rxFlowDetected * averAmp  
            rxSymbols_ = self.layerMerge(rxFlowDetected)
            rxSymbols = np.concatenate((rxSymbols, rxSymbols_))
        if supNoBits == 0:
            rxBits = self.demodulation(rxSymbols)
        else:
            rxBits = self.demodulation(rxSymbols)[:-supNoBits]
        return rxBits


if __name__ == "__main__":

    mimo_ofdm = MIMO_OFDM(2, 2, 64, "Perfect", "MMSE")



