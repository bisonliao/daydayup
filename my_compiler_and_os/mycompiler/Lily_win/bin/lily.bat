@IF "%1"==""  (echo "lily <script file name>"
goto endlabel)

prep %1 .lily.swp

lily_compile .lily.swp %1.out


:endlabel

