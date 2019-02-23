/*内核的一部分*/
extern void func_DispChr(char a);
extern void func_NewLine();
extern void func_DispHex(unsigned int i);
void func_C()
{
	char * p = (char*)0x0B8000;
	int line, col;

	for ( line = 0; line < 2; ++line)
		for ( col = 0; col < 20; ++col)
		{
			*(p+(80*line+col)*2) = '*';
			*(p+(80*line+col)*2+1) = 0x0c;
		}
	func_NewLine();
	func_NewLine();
	func_NewLine();
	func_DispChr('O');
	func_DispChr('K');
	func_NewLine();
	func_DispHex(0x01020304);
}
