function main
begin
	$count = 0;
	while $count < 100
	do
		$count = $count + 1;
		$A[$count] = "9";
	endwhile

	$count = 0;
	$sum = "";
	while $count < 100
	do
		$count = $count + 1;
		$sum = $sum + $A[$count] ;
	endwhile
	print($sum, "");
end
