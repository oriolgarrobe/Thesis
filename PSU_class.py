class PSU:
    def __init__(self, Vout, LS_Ron, Iout, Vin, HS_Tsw, Fsw, Vbody_diode, LS_QRR, DT, L, DCR, ESR_Cin, P_IC):
        self.Vout = Vout                  #Vpre_static -> SRD: UP3V3M = VPRE
        self.LS_Ron = LS_Ron              #Rds_on1 -> SRD: Q0700
        self.HS_Ron = LS_Ron              #Rds_on1 -> SRD: Q0700
        self.Iout = Iout                  #Iload_3V3M
        self.Vin = Vin                    #Vpre_in V -> INPUT!!!!
        self.HS_Tsw = HS_Tsw              #TSW_vpre [ns]
        self.LS_Tsw = HS_Tsw              #TSW_vpre [ns] 
        self.Fsw = Fsw                    #fSW_pre [kHz]
        self.Vbody_diode = Vbody_diode    #VForward_Voltage1
        self.LS_QRR = LS_QRR              #QRR_charge1 [nC]
        self.DTr = DT                     #DTrising [ns]
        self.DTf = DT                     #DTfalling [ns]
        self.L = L                        #LVPRE [microH]
        self.DCR = DCR                    #R_DC_L_VPRE_temp [mOhms]
        self.ESR_Cin = ESR_Cin            #ESR_3V3M_IN [mOhms]
        self.P_IC = P_IC                  # PMIC IC power dissipated [mW] P_Dissip_IC_3V3M
        
    def duty_cycle(self):
        """
        Duty Cycle -> D3v3
        """
        D = (self.Vout + (self.LS_Ron * self.Iout))/(self.Vin - (self.HS_Ron * self.Iout) + (self.LS_Ron * self.Iout))
        return D
    
    def HS_loss(self):
        """
        Conduction loss in the High-side MOSFET -> P_FET_3V3M_HS_cond_loss
        """
        D = self.duty_cycle()
        P_HS_cond = D * self.HS_Ron * self.Iout**2 #Watts
        return P_HS_cond
    
    def LS_loss(self):
        """
        Conduction loss in the Low-side MOSFET -> P_FET_3V3M_LS_cond_loss
        """
        D = self.duty_cycle()
        P_LS_cond = (1-D) * self.LS_Ron * self.Iout**2 #Watts
        return P_LS_cond
    
    def SW_HS_loss(self):
        """
        HS Switching-loss in the MOSFET -> PFET_3V3M_SW_HS_loss
        """
        P_HS_sw = (self.Vin * self.Iout) * self.HS_Tsw * self.Fsw #Watts
        return P_HS_sw
    
    def SW_LS_loss(self):
        """
        LS Body diode Reverse recovery-loss -> PFET_3V3M_SW_LS_loss
        """
        P_LS_sw = (self.Vbody_diode * self.Iout) * self.LS_Tsw * self.Fsw #Watts
        return P_LS_sw
    
    def RR_LS_loss(self):
        """
        LS Body diode Reverse recovery-loss -> PFET_3V3M_RR_LS_loss
        """
        P_Qrr = self.LS_QRR * self.Vin * self.Fsw / 2 #Watts
        return P_Qrr
    
    def DT_LS_loss(self):
        """
        LS Dead time-loss in the MOSFET body diode -> PFET_3V3M_DT_LS_loss
        """
        P_DT = self.Vbody_diode * self.Iout * (self.DTr + self.DTf) * self.Fsw #Watts
        return P_DT
    
    def AI_L(self):
        """
        Inductor Ripple current -> AI_L_3V3M
        """
        D = self.duty_cycle()
        AIL = (self.Vin - self.Vout - self.HS_Ron * self.Iout) * D / (self.Fsw * self.L) # Amperes
        return AIL
    
    def P_Dis_L(self):
        """
        Power Dissipated in the inductor -> P_Dis_L_3V3M
        """
        AIL = self.AI_L()
        P_L = self.DCR * (self.Iout**2 + AIL**2/12) #Watts
        return P_L
    
    def I_L_Peak(self):
        """
        Peak inductor current -> I_L_3V3M_PEAK
        """
        AIL = self.AI_L()
        Ipeak = self.Iout + AIL / 2
        return Ipeak
    
    def P_Dis_Cin(self):
        """
        Input capacitors Power Dissipated -> P_Dissip_Cin_3V3M
        """
        D = self.duty_cycle()
        I_Cin_rms = self.Iout * (D * ((1-D) + ((1-D) * self.HS_Tsw * self.Vout /(self.Iout * self.L))/12))**0.5
        P_Cin = self.ESR_Cin * I_Cin_rms**2
        return P_Cin
    
    def P_Dis_Tot(self):
        """
        Power Dissipated -> P_Dissip_3V3M_total Matrix
        """
        #Matrix
        P_HS_cond = self.HS_loss()
        P_LS_cond = self.LS_loss()
        P_HS_sw = self.SW_HS_loss()
        P_L = self.P_Dis_L()
        P_Qrr = self.RR_LS_loss()
        P_Cin = self.P_Dis_Cin()
        
        #Vector
        P_LS_sw = self.SW_LS_loss()
        P_DT = self.DT_LS_loss()
        
        #Total
        P_Dissip_Total = P_HS_cond + P_LS_cond + P_HS_sw + P_L + P_Qrr + P_Cin + P_LS_sw + P_DT + self.P_IC
        return P_Dissip_Total
    
    def P_out(self):
        """
        Power Output
        """
        return self.Iout * self.Vout
    
    def efficiency(self):
        """
        Efficiency of the Assembly -> n (nu)
        """
        Pout = self.P_out()
        Pdis = self.P_Dis_Tot()
        if (isinstance(Pout, int) or isinstance(Pout, float)):
            n = Pout / (Pout + Pdis)
        else:
            n = np.flip(Pout) / (np.flip(Pout) + np.flip(Pdis))
        return n
        
    def I_in(self):
        """
        Input battery -> I_IN_3V3M
        """
        n = self.efficiency()
        I_in = self.Iout * self.Vout / (n * self.Vin)
        return I_in
    
    def P_in(self):
        """
        Input battery -> P_IN_3V3M
        """
        Iin = self.I_in()
        P_in = Iin * self.Vin
        return P_in
    