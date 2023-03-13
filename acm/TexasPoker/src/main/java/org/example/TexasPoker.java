package org.example;

import java.security.SecureRandom;
import java.sql.SQLSyntaxErrorException;
import java.util.*;

public class TexasPoker {
    public List<Card> getCards() {
        return m_cards;
    }

    private List<Card> m_cards;

    static public final int VARIETY_DIAMOND = 0; // 方片
    static public final int VARIETY_HEART = 1; // 红桃
    static public final int VARIETY_CLUB = 2; // 梅花
    static public final int VARIETY_SPADE = 3; // 黑桃

    static public final int COMBINATION_ROYALFLUSH = 9; // 皇家同花顺
    static public final int COMBINATION_STRAIGHT_FLUSH = 8; // 同花顺
    static public final int COMBINATION_FOUR_OF_A_KIND = 7; //四条
    static public final int COMBINATION_FULL_HOUSE = 6; // 葫芦
    static public final int COMBINATION_FLUSH = 5; // 同花
    static public final int COMBINATION_STRAIGHT = 4; // 顺子
    static public final int COMBINATION_THREE_OF_A_KIND = 3; //三条
    static public final int COMBINATION_TOW_PAIR = 2; // 两对
    static public final int COMBINATION_PAIR = 1; // 一对
    static public final int COMBINATION_HIGH_CARD = 0; // 散牌



    static public List<Card> InitCardsAndShuffle()
    {
        List<Card> cards = new ArrayList<Card>();
        int i, j;
        for (i = 2; i < 15; ++i)
        {
            for (j = 0; j < 4; ++j)
            {
                Card c = new Card(i, j);
                cards.add(c);
            }
        }

        SecureRandom secureRandom = new SecureRandom();
        secureRandom.setSeed(System.currentTimeMillis());  //使用系统时间作为种子
        Collections.shuffle(cards, secureRandom);

        return cards;
    }
    static public void SortCards(List<Card> cards)
    {
        Collections.sort(cards, new Comparator<Card>() {
            @Override
            public int compare(Card o1, Card o2) {
                return  o1.getPoint() - o2.getPoint();
            }
        });
    }
    static public int JudgeCombination(List<Card> fiveCards)
    {
        return JudgeCombination(fiveCards.get(0), fiveCards.get(1),fiveCards.get(2),fiveCards.get(3),fiveCards.get(4));
    }
    static public int JudgeCombination(Card a, Card b, Card c, Card d, Card e)
    {
        boolean sameVariety = false; // 同花标志
        if (a.getVariety() == b.getVariety() &&
            b.getVariety() == c.getVariety() &&
            c.getVariety() == d.getVariety() &&
            d.getVariety() == e.getVariety()) // 同花
        {
            sameVariety = true;
        }
        List<Card> cards = new ArrayList<>();
        cards.add(a);
        cards.add(b);
        cards.add(c);
        cards.add(d);
        cards.add(e);
        SortCards(cards);
        int step = cards.get(1).getPoint() - cards.get(0).getPoint();
        int i;

        boolean ordinal = false; //顺子标志
        if (step == 1) { ordinal = true;}
        for (i = 1; i < 4; ++i)
        {
            if (cards.get(i+1).getPoint() - cards.get(i).getPoint() != step)
            {
                ordinal = false;
                break;
            }
        }
        boolean fourSame = false; //四条
        if (cards.get(0).getPoint() == cards.get(1).getPoint() &&
                cards.get(1).getPoint() == cards.get(2).getPoint() &&
                cards.get(2).getPoint() == cards.get(3).getPoint() ||
                cards.get(1).getPoint() == cards.get(2).getPoint() &&
                        cards.get(2).getPoint() == cards.get(3).getPoint() &&
                        cards.get(3).getPoint() == cards.get(4).getPoint()
        )
        {
            fourSame = true;
        }

        if (ordinal && sameVariety)
        {
            if (cards.get(4).getPoint() == 14)
            {
                return COMBINATION_ROYALFLUSH;
            }
            return COMBINATION_STRAIGHT_FLUSH;
        }
        if (fourSame)
        {
            return COMBINATION_FOUR_OF_A_KIND;
        }
        // fullHouse
        if (cards.get(0).getPoint() == cards.get(1).getPoint() &&
                cards.get(1).getPoint() == cards.get(2).getPoint() &&
                cards.get(3).getPoint() == cards.get(4).getPoint() ||
                cards.get(0).getPoint() == cards.get(1).getPoint() &&
                        cards.get(2).getPoint() == cards.get(3).getPoint() &&
                        cards.get(3).getPoint() == cards.get(4).getPoint()
        )
        {
            return COMBINATION_FULL_HOUSE;
        }
        if (sameVariety)
        {
            return COMBINATION_FLUSH;
        }
        if (ordinal)
        {
            return COMBINATION_STRAIGHT;
        }
        // 三条
        if (cards.get(0).getPoint() == cards.get(1).getPoint() &&  cards.get(1).getPoint() == cards.get(2).getPoint() ||
                cards.get(3).getPoint() == cards.get(4).getPoint()  &&  cards.get(2).getPoint() == cards.get(3).getPoint())
        {
            return COMBINATION_THREE_OF_A_KIND;
        }
        int pairNum = 0;
        for (i = 0; i < cards.size()-1; ++i)
        {
            if (cards.get(i).getPoint() == cards.get(i+1).getPoint())
            {
                pairNum++;
                i++;
            }
        }
        if (pairNum == 2)
        {
            return COMBINATION_TOW_PAIR;
        }
        if (pairNum == 1)
        {
            return COMBINATION_PAIR;
        }
        return COMBINATION_HIGH_CARD;
    }

    public TexasPoker()
    {
        m_cards = InitCardsAndShuffle();
    }
    private void removeCardFromList(Card a)
    {
        int i;
        for (i = 0; i < m_cards.size(); i++)
        {
            if (m_cards.get(i).getPoint() == a.getPoint() &&
                m_cards.get(i).getVariety() == a.getVariety())
            {
                m_cards.remove(i);
                break;
            }
        }
    }
    public void ThreeCardOnDest(Card a, Card b, Card c)
    {
        removeCardFromList(a);
        removeCardFromList(b);
        removeCardFromList(c);
        int i, j;
        int totalNum = 0;
        int count[] = new int[10]; // ten combinations
        for (i = 0; i < 10; ++i)
        {
            count[i] = 0;
        }
        for (i = 0; i < m_cards.size(); ++i)
        {
            for (j = 0; j < m_cards.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                Card d = m_cards.get(i);
                Card e = m_cards.get(j);
                int combination = JudgeCombination(a, b, c, d, e);
                count[combination]++;
                totalNum++;
            }
        }
        if (totalNum == 0)
        {
            return;
        }
        System.out.println(""+a.getPoint()+","+a.getVariety());
        System.out.println(""+b.getPoint()+","+b.getVariety());
        System.out.println(""+c.getPoint()+","+c.getVariety());
        System.out.println("--------------"+totalNum);


        System.out.println("皇家同花顺: " + (count[9]* 1.0 / totalNum));
        System.out.println("同花顺：" + (count[8]* 1.0 / totalNum));
        System.out.println("四条：" + (count[7]* 1.0 / totalNum));
        System.out.println("葫芦：" + (count[6]* 1.0 / totalNum));
        System.out.println("同花：" + (count[5]* 1.0 / totalNum));
        System.out.println("顺子：" + (count[4]* 1.0 / totalNum));
        System.out.println("三条：" + (count[3]* 1.0 / totalNum));
        System.out.println("两对：" + (count[2]* 1.0 / totalNum));
        System.out.println("一对：" + (count[1]* 1.0 / totalNum));
        System.out.println("散户：" + (count[0]* 1.0 / totalNum));

    }
    public void FourCardOnDest(Card a, Card b, Card c, Card d)
    {
        removeCardFromList(a);
        removeCardFromList(b);
        removeCardFromList(c);
        removeCardFromList(d);
        int i, j;
        int totalNum = 0;
        int count[] = new int[10]; // ten combinations
        for (i = 0; i < 10; ++i)
        {
            count[i] = 0;
        }
        for (i = 0; i < m_cards.size(); ++i)
        {
            for (j = 0; j < m_cards.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                Card f = m_cards.get(i);
                Card e = m_cards.get(j);

                int maxCombination = -1;


                int p;
                for (p = 0; p < 6; ++p)
                {
                    List<Card> sixCards = new ArrayList<>();
                    sixCards.add(a);
                    sixCards.add(b);
                    sixCards.add(c);
                    sixCards.add(d);
                    sixCards.add(e);
                    sixCards.add(f);

                    sixCards.remove(p);
                    int combination = JudgeCombination(sixCards);
                    if (combination > maxCombination)
                    {
                        maxCombination = combination;
                    }

                }
                count[maxCombination]++;
                totalNum++;


            }
        }
        if (totalNum == 0)
        {
            return;
        }
        System.out.println(""+a.getPoint()+","+a.getVariety());
        System.out.println(""+b.getPoint()+","+b.getVariety());
        System.out.println(""+c.getPoint()+","+c.getVariety());
        System.out.println(""+d.getPoint()+","+d.getVariety());
        System.out.println("--------------"+totalNum);


        System.out.println("皇家同花顺: " + (count[9]* 1.0 / totalNum));
        System.out.println("同花顺：" + (count[8]* 1.0 / totalNum));
        System.out.println("四条：" + (count[7]* 1.0 / totalNum));
        System.out.println("葫芦：" + (count[6]* 1.0 / totalNum));
        System.out.println("同花：" + (count[5]* 1.0 / totalNum));
        System.out.println("顺子：" + (count[4]* 1.0 / totalNum));
        System.out.println("三条：" + (count[3]* 1.0 / totalNum));
        System.out.println("两对：" + (count[2]* 1.0 / totalNum));
        System.out.println("一对：" + (count[1]* 1.0 / totalNum));
        System.out.println("散户：" + (count[0]* 1.0 / totalNum));

    }
    public void FiveCardOnDest(Card a, Card b, Card c, Card d, Card e)
    {
        removeCardFromList(a);
        removeCardFromList(b);
        removeCardFromList(c);
        removeCardFromList(d);
        removeCardFromList(e);
        int i, j;
        int totalNum = 0;
        int count[] = new int[10]; // ten combinations
        for (i = 0; i < 10; ++i)
        {
            count[i] = 0;
        }

        for (i = 0; i < m_cards.size(); ++i)
        {
            for (j = 0; j < m_cards.size(); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                Card f = m_cards.get(i);
                Card g = m_cards.get(j);

                int maxCombination = -1;


                int p,q;
                for (p = 0; p < 7; ++p)
                {
                    List<Card> sevenCards = new ArrayList<>();
                    sevenCards.add(a);
                    sevenCards.add(b);
                    sevenCards.add(c);
                    sevenCards.add(d);
                    sevenCards.add(e);
                    sevenCards.add(f);
                    sevenCards.add(g);

                    sevenCards.remove(p);
                    for (q = 0; q < sevenCards.size();++q)
                    {
                        List<Card> sixCards = new ArrayList<>();
                        sixCards.add(sevenCards.get(0));
                        sixCards.add(sevenCards.get(1));
                        sixCards.add(sevenCards.get(2));
                        sixCards.add(sevenCards.get(3));
                        sixCards.add(sevenCards.get(4));
                        sixCards.add(sevenCards.get(5));

                        sixCards.remove(q);
                        int combination = JudgeCombination(sixCards);
                        if (combination > maxCombination)
                        {
                            maxCombination = combination;
                        }

                        /*
                        if (combination > 1)
                        {
                            System.out.println("" + sixCards.get(0).getPoint() + "," + sixCards.get(0).getVariety());
                            System.out.println("" + sixCards.get(1).getPoint() + "," + sixCards.get(1).getVariety());
                            System.out.println("" + sixCards.get(2).getPoint() + "," + sixCards.get(2).getVariety());
                            System.out.println("" + sixCards.get(3).getPoint() + "," + sixCards.get(3).getVariety());
                            System.out.println("" + sixCards.get(4).getPoint() + "," + sixCards.get(4).getVariety());
                            System.out.println("combination:" + combination);

                        }

                         */

                    }

                }
                count[maxCombination]++;
                totalNum++;
            }
        }
        if (totalNum == 0)
        {
            return;
        }

        System.out.println(""+a.getPoint()+","+a.getVariety());
        System.out.println(""+b.getPoint()+","+b.getVariety());
        System.out.println(""+c.getPoint()+","+c.getVariety());
        System.out.println(""+d.getPoint()+","+d.getVariety());
        System.out.println(""+e.getPoint()+","+e.getVariety());
        System.out.println("--------------"+totalNum);


        System.out.println("皇家同花顺: " + (count[9]* 1.0 / totalNum));
        System.out.println("同花顺：" + (count[8]* 1.0 / totalNum));
        System.out.println("四条：" + (count[7]* 1.0 / totalNum));
        System.out.println("葫芦：" + (count[6]* 1.0 / totalNum));
        System.out.println("同花：" + (count[5]* 1.0 / totalNum));
        System.out.println("顺子：" + (count[4]* 1.0 / totalNum));
        System.out.println("三条：" + (count[3]* 1.0 / totalNum));
        System.out.println("两对：" + (count[2]* 1.0 / totalNum));
        System.out.println("一对：" + (count[1]* 1.0 / totalNum));
        System.out.println("散户：" + (count[0]* 1.0 / totalNum));

    }
}
