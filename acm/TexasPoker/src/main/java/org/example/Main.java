package org.example;



public class Main {
    public static void main(String[] args)
    {
        TexasPoker poker = new TexasPoker();
        Card a = poker.getCards().get(0);
        Card b = poker.getCards().get(1);
        Card c = poker.getCards().get(2);
       // Card d = poker.getCards().get(3);
       // Card e = poker.getCards().get(4);
        poker.ThreeCardOnDest(a, b, c);
    }
}