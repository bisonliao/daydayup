#include "rtl8139.h"
#include "global.h"
#include "packet.h"
#include "types.h"
#include "redefine.h"

#define interrupt 
#define far

void interrupt NewFunction(void);
int	INTR;
int directvideo;


void InitHardware();
void InitSoftware();
void IssueCMD(unsigned char descriptor);
void TxInterruptHandler();
int RxInterruptHandler();

#define	RX_BUFFER_SIZE		16*1024
#define RX_MAX_PACKET_LENGTH	1600
#define RX_MIN_PACKET_LENGTH	64
#define	RX_READ_POINTER_MASK	0x3FFC

//Transmit variables
#define TX_SW_BUFFER_NUM	4
TX_DESCRIPTOR	TxDesc[TX_SW_BUFFER_NUM];
unsigned char   TxBuffer[TX_SW_BUFFER_NUM][ 1600 / sizeof(int) ];
unsigned char	TxHwSetupPtr;
unsigned char	TxHwFinishPtr;
unsigned char	TxHwFreeDesc;

//Receive variables
unsigned char   RxSpace[ RX_BUFFER_SIZE + 2000 ];
unsigned char	*RxBuffer,*RxBufferOriginal;
unsigned long	RxBufferPhysicalAddress;
unsigned int	RxReadPtrOffset;
PPACKET		pLeadingReadPacket;	//should be a link list
unsigned long	PacketReceivedGood = 0;
unsigned long	ByteReceived = 0;

uint32_t	IOBase,Irq;
unsigned char *Buffer;
uint32_t	PhysicalAddrBuffer;



int FindIOIRQ(uint32_t *IOBase,uint32_t *IRQ)
{
    uint32_t i,j,PciData;
    for(i=0;i<32;i++)
    {
        j=0x80000000+(i<<11);
        out_word(0xcf8,j);
        PciData=in_byte(0xcfc);
        if(PciData==0x813910ec)
        {
            out_word(0xcf8,j+0x10);
            *IOBase=in_byte(0xcfc);
            *IOBase &= 0xfffffff0;
            out_word(0xcf8,j+0x3c);
            *IRQ=in_byte(0xcfc);
            *IRQ &= 0xff;
            return TRUE;
        }
    }
    return FALSE;
}


int ComputeInterrupt( int	IrqNumber)
{
    if(IrqNumber <=8) return IrqNumber+8;
    else		  return IrqNumber+0x68;
}
/////////////////////////////////////////////////////////////////////////
//Our Interrupt Service Routine (ISR)
/////////////////////////////////////////////////////////////////////////
void interrupt NewFunction(void)
{
    unsigned int curISR;
    printk("nic interrupted!\n");
    __asm__("cli");
    curISR = in_word(IOBase + ISR);
    if( (curISR & R39_INTERRUPT_MASK) != 0)
    {
        do
        {
            if(curISR & ISR_PUN)
            {
                //     	        ProcessLingChange();	//should write this code someday
                out_byte(IOBase + ISR , ISR_PUN);
            }
            if(curISR & ISR_TOK)
            {
                TxInterruptHandler();
                out_byte(IOBase + ISR, ISR_TOK);
            }
            if(curISR & ISR_TER)
            {
                out_byte(IOBase + TCR , TCR_CLRABT);
                out_byte(IOBase + ISR , ISR_TER);
            }
            if( curISR & (ISR_ROK|ISR_RER|ISR_RXOVW|ISR_FIFOOVW) )
            {
                if(curISR & ISR_ROK)
                {
                    RxInterruptHandler();
                }
                out_byte(IOBase + ISR, ISR_ROK | ISR_RER |ISR_RXOVW | ISR_FIFOOVW);
            }
            curISR = in_word(IOBase + ISR);
        }while( (curISR & R39_INTERRUPT_MASK) != 0);
        //	_asm int 3;
        /*
        _asm    mov     al,020h
            _asm    out     0a0h,al         //;issue EOI to 2nd 8259
            _asm    out     20h,al          //;Issue EOI to 1nd 8259
        */
        out_byte(0xa0, 0x20);
        out_byte(0x20, 0x20);
    }
    else
    {//not our interrupt, should call original interrupt service routine.
        // OldFunction();
    }
    __asm__("sti");
}

//////////////////////////////////////////////////////////////////////////
//Initialization part
//////////////////////////////////////////////////////////////////////////
void InitHardware()
{
    out_byte(IOBase + CR, CR_RST);              //reset
    out_byte(IOBase + CR, CR_RE + CR_TE);       //enable Tx/Rx
    out_word(IOBase + TCR, 	TCR_IFG0   |
            TCR_IFG1   |
            TCR_MXDMA2 |
            TCR_MXDMA1);

    out_word(IOBase + RCR, 	RCR_RBLEN0 |
            RCR_MXDMA2 |
            RCR_MXDMA1 |
            RCR_AB	   |
            RCR_AM     |
            RCR_APM );
    out_word(IOBase + RBSTART , RxBufferPhysicalAddress);
    out_byte(IOBase + IMR, R39_INTERRUPT_MASK);//enable interrupt
}

void InitSoftware()
{
    uint32_t	Offset,Segment,Delta,i;
    unsigned char *tmpBuffer;
    //Init Tx Variables
    TxHwSetupPtr = 0;
    TxHwFinishPtr    = 0;
    TxHwFreeDesc = TX_SW_BUFFER_NUM;
    //initialize TX descriptor
    for(i=0;i<TX_SW_BUFFER_NUM;i++)
    {	//allocate memory
        Buffer=TxBuffer[i];
        TxDesc[i].OriginalBufferAddress = Buffer;
        Offset=(uint32_t)(Buffer);
        //align to DWORD
        if( Offset & 0x3 )
        {
            Delta = 4 - (Offset & 0x3);
            Offset = Offset + Delta;
            Buffer = Buffer + Delta;
        }
        TxDesc[i].buffer = Buffer;
        TxDesc[i].PhysicalAddress = (uint32_t)Buffer;
        //		TxDesc[i].DescriptorStatus = TXDESC_INIT;
    }
    //Init Rx Buffer
    RxBufferOriginal =
        tmpBuffer	 = RxSpace;
    Offset=(uint32_t)(tmpBuffer);
    //align to DWORD
    if( Offset & 0x3 )
    {
        Delta = 4 - (Offset & 0x3);
        Offset = Offset + Delta;
        tmpBuffer = tmpBuffer + Delta;
    }
    RxBuffer = tmpBuffer;
    RxBufferPhysicalAddress = (uint32_t)tmpBuffer;
    //Init Rx Variable
    RxReadPtrOffset = 0;
}

//////////////////////////////////////////////////////////////////////////
//Transmit part
//////////////////////////////////////////////////////////////////////////
unsigned char NextDesc( unsigned char CurrentDescriptor)
{
    //    (CurrentDescriptor == TX_SW_BUFFER_NUM-1) ? 0 : (1 + CurrentDescriptor);
    if(CurrentDescriptor == TX_SW_BUFFER_NUM-1)
    {
        return  0;
    }
    else
    {
        return ( 1 + CurrentDescriptor);
    }
}

unsigned char CheckTSDStatus( unsigned char            Desc)
{
    uint32_t       Offset = Desc << 2;
    uint32_t       tmpTSD;

    tmpTSD=in_byte(IOBase + TSD0 + Offset);
    switch ( tmpTSD & (TSD_OWN | TSD_TOK) )
    {
        case (TSD_OWN | TSD_TOK):      	return 	TSDSTATUS_BOTH;
        case (TSD_TOK) 		:       return  TSDSTATUS_TOK;
        case (TSD_OWN) 		:       return  TSDSTATUS_OWN;
        case 0 			:	return  TSDSTATUS_0;
    }
    return 0;
}



void IssueCMD(unsigned char descriptor)
{
    unsigned long offset = descriptor << 2; // ??这里有问题吧
    /*
    printk("write to %u\n",IOBase + TSAD0 + offset);
    printk("write to %u\n",IOBase + TSD0 + offset);
    */
    out_word(IOBase + TSAD0 + offset, TxDesc[TxHwSetupPtr].PhysicalAddress);
    out_word(IOBase + TSD0 + offset , TxDesc[TxHwSetupPtr].PacketLength);
}

int SendPacket( PPACKET pPacket)
{
    __asm__("cli");
    if( TxHwFreeDesc>0  )
    {
        TxDesc[TxHwSetupPtr].PacketLength=
            CopyFromPacketToBuffer( pPacket , TxDesc[TxHwSetupPtr].buffer);
        IssueCMD(TxHwSetupPtr);
        TxHwSetupPtr = NextDesc(TxHwSetupPtr);
        TxHwFreeDesc--;
        __asm__("sti");
        return TRUE;//success
    }
    else
    {
        __asm__("sti");
        return FALSE;//out of resource
    }
}
int SendPacket2()
{
    __asm__("cli");
    IssueCMD(0);
    __asm__("sti");
    return TRUE;//success
}

    void
TxInterruptHandler()
{
    while( (CheckTSDStatus(TxHwFinishPtr) == TSDSTATUS_BOTH	) &&
            (TxHwFreeDesc < 4 				)   )
    {
        //can Release this buffer now

        TxHwFinishPtr = NextDesc(TxHwFinishPtr);
        TxHwFreeDesc++;
    }
}
////////////////////////////////////////////////////////////////////////
// Start of Rx Path
////////////////////////////////////////////////////////////////////////
void
ReadPacket(
        PPACKET	RxPacket
        )
{
    pLeadingReadPacket = RxPacket;
}

void
CopyPacket(
        unsigned char 	*pIncomePacket,
        unsigned int        PktLength
        )
{
    if( (pLeadingReadPacket != NULL)           &&
            (pLeadingReadPacket->PacketLength == 0)  )
    {
        memcpy(pLeadingReadPacket->Buffers.Buffer , pIncomePacket , PktLength);
        pLeadingReadPacket->PacketLength = PktLength;
    }

}

int
PacketOK(
        PPACKETHEADER pPktHdr
        )
{
    int BadPacket = pPktHdr->RUNT ||
        pPktHdr->LONG ||
        pPktHdr->CRC  ||
        pPktHdr->FAE;
    if( ( !BadPacket )   &&
            ( pPktHdr->ROK )   )
    {
        if ( (pPktHdr->PacketLength > RX_MAX_PACKET_LENGTH ) ||
                (pPktHdr->PacketLength < RX_MIN_PACKET_LENGTH )    )
        {
            return(FALSE);
        }
        PacketReceivedGood++;
        ByteReceived += pPktHdr->PacketLength;
        return TRUE ;
    }
    else
    {
        return FALSE;
    }
}

int
RxInterruptHandler(
        )
{
    unsigned char  TmpCMD;
    unsigned int   PktLength;
    unsigned char  *pIncomePacket, *RxReadPtr;
    PPACKETHEADER  pPacketHeader;

    while (TRUE)
    {
        TmpCMD = in_byte(IOBase + CR);
        if (TmpCMD & CR_BUFE)
        {
            break;
        }

        do
        {
            RxReadPtr	  = RxBuffer + RxReadPtrOffset;
            pPacketHeader = (PPACKETHEADER)  RxReadPtr;
            pIncomePacket = RxReadPtr + 4;
            PktLength	  = pPacketHeader->PacketLength;	//this length include CRC
            if ( PacketOK( pPacketHeader ) )
            {
                if ( (RxReadPtrOffset + PktLength) > RX_BUFFER_SIZE )
                {      //wrap around to end of RxBuffer
                    //_asm int 3;
                    memcpy( RxBuffer + RX_BUFFER_SIZE ,	RxBuffer,
                            (RxReadPtrOffset + PktLength - RX_BUFFER_SIZE)  );
                }
                //copy the packet out here
                CopyPacket(pIncomePacket,PktLength - 4);//don't copy 4 bytes CRC

                //update Read Pointer
                RxReadPtrOffset = (RxReadPtrOffset + PktLength + 4 + 3) & RX_READ_POINTER_MASK;
                //4:for header length(PktLength include 4 bytes CRC)
                //3:for dword alignment
                out_byte( IOBase + CAPR, RxReadPtrOffset - 0x10);	//-4:avoid overflow
            }
            else
            {
                //		ResetRx();
                break;
            }
            TmpCMD = in_byte(IOBase + CR);
        } while (!(TmpCMD & CR_BUFE));
    }
    return (TRUE);              //Done
}

/////////////////////////////////////////////////////////////////////////
//Load / Unload
/////////////////////////////////////////////////////////////////////////
    int
LoadDriver()
{
    int	INTR;
    uint8_t mask;

    FindIOIRQ( &IOBase , &Irq );
    INTR = ComputeInterrupt( Irq );
    printk("IOBase=%u, Irq=%u, INTR=%u\n", IOBase, Irq, INTR);
    //hook interrupt vector
    //initialize chip
    InitSoftware();
    InitHardware();

    /*打开中断屏蔽*/
    mask = in_byte(INT_S_CTLMASK); 
    mask = mask & 0xf8;
    out_byte(INT_S_CTLMASK, mask);

    return TRUE;
}

    int
UnloadDriver()
{
    return TRUE;
}
///////////////////////////////////////////////////////////////////////////
// End Of Demo driver
///////////////////////////////////////////////////////////////////////////

