function main
begin
	$xxx = 100;
	print("$xxx=", $xxx, "\n");
	print("$xxx * $xxx = ", double($xxx), "\n"); 
end

function double
begin
	return $1*$1;
end
