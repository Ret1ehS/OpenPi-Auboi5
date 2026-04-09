$ServerHost = "__SERVER_HOST__"
$ServerPort = __SERVER_PORT__
$Token = "__TOKEN__"
$PollMs = 10

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class OpenPiKeyboardNative {
    [DllImport("user32.dll")]
    public static extern short GetAsyncKeyState(Int32 vKey);
}
"@

function Test-KeyDown([int]$VirtualKey) {
    return (([OpenPiKeyboardNative]::GetAsyncKeyState($VirtualKey) -band 0x8000) -ne 0)
}

function Write-JsonLine([System.IO.StreamWriter]$Writer, [string]$Line) {
    $Writer.WriteLine($Line)
    $Writer.Flush()
}

while ($true) {
    $client = $null
    $writer = $null
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.NoDelay = $true
        $client.Connect($ServerHost, $ServerPort)
        $stream = $client.GetStream()
        $writer = New-Object System.IO.StreamWriter($stream, [System.Text.Encoding]::UTF8)
        $writer.NewLine = "`n"
        $writer.AutoFlush = $true

        $prevEnter = $false
        $prevSpace = $false
        $prevShift = $false
        $prevTab = $false
        $prevQ = $false
        $prevEsc = $false

        while ($client.Connected) {
            $up = Test-KeyDown 0x26
            $down = Test-KeyDown 0x28
            $left = Test-KeyDown 0x25
            $right = Test-KeyDown 0x27
            $ctrl = Test-KeyDown 0x11
            $shift = Test-KeyDown 0x10
            $tab = Test-KeyDown 0x09

            Write-JsonLine $writer ("{""type"":""state"",""token"":""" + $Token + """,""up"":" + $up.ToString().ToLower() + ",""down"":" + $down.ToString().ToLower() + ",""left"":" + $left.ToString().ToLower() + ",""right"":" + $right.ToString().ToLower() + ",""ctrl"":" + $ctrl.ToString().ToLower() + "}")

            $enterNow = Test-KeyDown 0x0D
            if ($enterNow -and -not $prevEnter) {
                Write-JsonLine $writer ("{""type"":""event"",""token"":""" + $Token + """,""key"":""ENTER""}")
            }
            $prevEnter = $enterNow

            $spaceNow = Test-KeyDown 0x20
            if ($spaceNow -and -not $prevSpace) {
                Write-JsonLine $writer ("{""type"":""event"",""token"":""" + $Token + """,""key"":""SPACE""}")
            }
            $prevSpace = $spaceNow

            if ($shift -and -not $prevShift) {
                Write-JsonLine $writer ("{""type"":""event"",""token"":""" + $Token + """,""key"":""SHIFT""}")
            }
            $prevShift = $shift

            if ($tab -and -not $prevTab) {
                Write-JsonLine $writer ("{""type"":""event"",""token"":""" + $Token + """,""key"":""TAB""}")
            }
            $prevTab = $tab

            $qNow = Test-KeyDown 0x51
            if ($qNow -and -not $prevQ) {
                Write-JsonLine $writer ("{""type"":""event"",""token"":""" + $Token + """,""key"":""QUIT""}")
            }
            $prevQ = $qNow

            $escNow = Test-KeyDown 0x1B
            if ($escNow -and -not $prevEsc) {
                Write-JsonLine $writer ("{""type"":""event"",""token"":""" + $Token + """,""key"":""QUIT""}")
                break
            }
            $prevEsc = $escNow

            Start-Sleep -Milliseconds $PollMs
        }
    } catch {
        Start-Sleep -Milliseconds 500
    } finally {
        if ($writer -ne $null) {
            try { $writer.Dispose() } catch {}
        }
        if ($client -ne $null) {
            try { $client.Close() } catch {}
        }
    }
    Start-Sleep -Milliseconds 250
}
