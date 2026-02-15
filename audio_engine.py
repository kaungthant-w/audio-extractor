_K='status'
_J='instrument'
_I='match_input'
_H='balanced'
_G='.mp3'
_F='.wav'
_E='sos'
_D=None
_C=1.
_B=False
_A=True
import sys,os,json,subprocess,numpy as np,librosa,soundfile as sf,torch,scipy.signal
FFMPEG_PATH=_D
try:
	import imageio_ffmpeg;FFMPEG_PATH=imageio_ffmpeg.get_ffmpeg_exe()
	if not os.path.exists(FFMPEG_PATH):FFMPEG_PATH=_D
except ImportError:pass
def convert_to_wav(input_path,output_wav_path):
	if not FFMPEG_PATH:raise RuntimeError('ffmpeg is required to convert non-WAV files.')
	A=[FFMPEG_PATH,'-y','-i',input_path,'-ar','44100','-ac','2','-f','wav',output_wav_path];subprocess.run(A,capture_output=_A,text=_A,timeout=120)
def convert_wav_to_mp3(input_wav_path,output_mp3_path):
	if not FFMPEG_PATH:return _B
	A=[FFMPEG_PATH,'-y','-i',input_wav_path,'-codec:a','libmp3lame','-b:a','192k',output_mp3_path];B=subprocess.run(A,capture_output=_A,text=_A,timeout=120);return B.returncode==0
def normalize_audio(audio_array,target_vol=.85):
	A=audio_array;B=np.max(np.abs(A))
	if B>0:return A/B*target_vol
	return A
def apply_auto_eq(audio,sr,mode,strength_factor):
	"\n    Apply EQ presets: balanced, warm, bright.\n    strength_factor: 0.0 to 1.0 (controls mix of EQ'd signal)\n    ";D=strength_factor;B=mode;A=audio
	if B==_H:C=scipy.signal.butter(2,[300,4000],btype='bandstop',fs=sr,output=_E)
	elif B=='warm':C=scipy.signal.butter(2,400,btype='lowshelf',fs=sr,output=_E)
	elif B=='bright':C=scipy.signal.butter(2,2000,btype='highshelf',fs=sr,output=_E)
	elif B==_I:return A
	else:return A
	E=scipy.signal.sosfilt(C,A);return(1-D)*A+D*E
def separate_with_demucs(audio_path,output_dir):
	J='vocals';D=output_dir;from demucs.pretrained import get_model as K;from demucs.apply import apply_model as L;print('  Loading Demucs AI model...',flush=_A);A=K('htdemucs');A.eval();print('  Loading audio...',flush=_A);C,S=librosa.load(audio_path,sr=A.samplerate,mono=_B)
	if C.ndim==1:C=np.stack([C,C])
	M=torch.tensor(C,dtype=torch.float32).unsqueeze(0);print('  Running AI separation...',flush=_A)
	with torch.no_grad():B=L(A,M,device='cpu')[0]
	if isinstance(B,torch.Tensor):B=B.cpu().numpy()
	N=A.sources.index(J);O=A.sources.index('other');E=B[N];P=B[O];F=np.zeros_like(E)
	for(Q,R)in enumerate(A.sources):
		if R!=J:F+=B[Q]
	G=os.path.join(D,'vocals.wav');H=os.path.join(D,'accompaniment.wav');I=os.path.join(D,'other.wav');sf.write(G,E.T,A.samplerate);sf.write(H,F.T,A.samplerate);sf.write(I,P.T,A.samplerate);return G,H,I
def synthesize_instrument(vocals_path,instrument,effects_opts):
	V=vocals_path;R=.0;L=instrument;D=effects_opts;print('  Loading vocals for pitch detection...',flush=_A);E,F=librosa.load(V,sr=22050,mono=_A);t=float(D.get('glitch_threshold',.08));u=float(D.get('amp_smoothing',15));v=float(D.get('pitch_smoothing',5));e=float(D.get('reverb_amount',.15))
	if D.get('preview',_B):
		print('  [Preview Mode] Trimming analysis to 10 seconds...',flush=_A);f=10*F
		if len(E)>f:E=E[:f]
	if D.get('cleanup_reverb',_B):print('  [Cleanup] Removing Reverb tail...',flush=_A);w=np.max(np.abs(E))*.01;E[np.abs(E)<w]=0
	try:from basic_pitch.inference import predict as x;S=_A;print('  Detecting precise notes with AI Basic Pitch...',flush=_A)
	except ImportError:S=_B;print("  [Warning] 'basic-pitch' library not found. Falling back to librosa.pyin.",flush=_A);print('  To use AI pitch detection, run: pip install basic-pitch',flush=_A)
	M=512;N=len(E)//M+1;B=np.zeros(N);I=np.zeros(N)
	if S:
		try:
			g=V;print(f"  Detecting precise notes with AI Basic Pitch from {g}...",flush=_A);AA,y,z=x(g);h=V.replace(_F,'_melody.mid')
			try:y.write(h);print(f"  [Success] MIDI file saved to: {h}",flush=_A)
			except Exception as W:print(f"  [Warning] Could not save MIDI file: {W}",flush=_A)
			i=np.zeros(N)
			for T in z:
				j=T[0];k=T[1];A0=T[2];X=T[3]
				if k-j<t:continue
				A1=int(j*F/M);Y=int(k*F/M);A2=librosa.midi_to_hz(A0);Y=min(Y,N)
				for A in range(A1,Y):
					if A<N:
						if X>i[A]:B[A]=A2;I[A]=X;i[A]=X
		except Exception as W:print(f"  [Error] Basic Pitch failed: {W}. Falling back to librosa.",flush=_A);S=_B
	if not S:print('  Detecting pitch with librosa...',flush=_A);l,AB,A3=librosa.pyin(E,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'),sr=F,frame_length=2048,hop_length=512);J=min(len(l),N);B[:J]=np.nan_to_num(l[:J]);I[:J]=np.nan_to_num(A3[:J])
	B=np.nan_to_num(B);I=np.nan_to_num(I);Z=0
	for U in range(len(B)):
		if B[U]<=0 and Z>0:B[U]=Z
		elif B[U]>0:Z=B[U]
	if D.get('cleanup_harmonies',_B):print('  [Cleanup] Smoothing pitch to remove harmonies...',flush=_A);B=scipy.signal.medfilt(B,kernel_size=7)
	else:B=scipy.signal.medfilt(B,kernel_size=3)
	print(f"  Synthesizing {L}...",flush=_A);m=D.get('vibrato',_B);n=float(D.get('strength',50))/1e2;A4=5.;A5=.02*n if m else R;Q=len(E);G=np.zeros(Q,dtype=np.float32);M=512;H=librosa.feature.rms(y=E,frame_length=2048,hop_length=M)[0]
	if np.max(H)>0:H/=np.max(H)
	J=min(len(B),len(H),len(I));B=B[:J];H=H[:J];I=I[:J];A6=.6;H[I<A6]=0;o=np.arange(len(H))*M;p=np.arange(Q);K=np.interp(p,o,H);A7=.9997
	for A in range(1,Q):
		if K[A]<K[A-1]:K[A]=max(K[A],K[A-1]*A7)
	from scipy.ndimage import gaussian_filter1d as q;K=q(K,sigma=u);a=np.interp(p,o,B);a=q(a,sigma=v);C=R
	for A in range(Q):
		b=a[A];r=K[A]
		if r<.001:G[A]=R;continue
		if b>0:
			if m:b*=_C+A5*np.sin(2*np.pi*A4*A/F)
			C+=2.*np.pi*b/F
			if L=='flute':A8=np.random.normal(0,.01);O=_C*np.sin(C)+.05*np.sin(2*C)+.12*np.sin(3*C)+A8
			elif L in('pianet','vintage_organ'):c=C/(2*np.pi)%_C;O=.75*(2*np.abs(2*c-1)-1)+.25*np.sin(C)
			elif L=='cello':c=C/(2*np.pi)%_C;O=.65*(2*c-1)+.35*np.sin(C)
			elif L=='guitar':O=.5*np.sin(C)+.25*np.sin(2*C)+.15*np.sin(3*C)
			elif L=='saxophone':O=np.clip(np.sin(C)+.33*np.sin(3*C),-1,1)
			else:O=np.sin(C)
			G[A]=O*r*.5
		else:G[A]=R;C=R
	print('  [Effect] Applying Reverb...',flush=_A);s=int(F*1.5);A9=np.random.uniform(-1,1,s)*np.exp(-np.linspace(0,5,s));P=scipy.signal.fftconvolve(G,A9,mode='full');P=P[:Q]
	if np.max(np.abs(P))>0:P/=np.max(np.abs(P))
	G=(1-e)*G+e*P*.6;d=D.get('eq',_H)
	if d!=_I:print(f"  [Effect] Applying Auto EQ: {d}...",flush=_A);G=apply_auto_eq(G,F,d,n)
	return normalize_audio(G),F
def dynamic_mix(vocals_path,accompaniment_path,instrument_audio,inst_sr,mix_options,output_path,mix_balance=.5):
	H=mix_options;G=inst_sr;D=mix_balance;C=instrument_audio;A=output_path;I,N=librosa.load(vocals_path,sr=G,mono=_A);J,N=librosa.load(accompaniment_path,sr=G,mono=_A);K=[len(I),len(J)]
	if C is not _D:K.append(len(C))
	E=min(K);B=np.zeros(E,dtype=np.float32);L=_C;M=_C
	if D<.5:M=D*2.
	elif D>.5:L=(_C-D)*2.
	if'vocal'in H:B+=normalize_audio(I[:E],.9*L)
	if'music'in H:B+=normalize_audio(J[:E],.8)
	if _J in H and C is not _D:B+=normalize_audio(C[:E],.9*M)
	B=normalize_audio(B,.95);F=A.rsplit('.',1)[0]+'_temp.wav';sf.write(F,B,G)
	if FFMPEG_PATH and A.endswith(_G):convert_wav_to_mp3(F,A);os.remove(F)
	else:
		if A.endswith(_G):A=A.rsplit('.',1)[0]+_F
		os.replace(F,A)
	return A
if __name__=='__main__':
	input_file=sys.argv[1];instrument=sys.argv[2];job_id=sys.argv[3];mix_options_str=sys.argv[4]if len(sys.argv)>4 else'vocal,instrument';effects_file=sys.argv[5]if len(sys.argv)>5 else _D;effects_opts={}
	if effects_file and os.path.exists(effects_file):
		with open(effects_file,'r')as f:effects_opts=json.load(f)
	mix_options_list=[A.strip()for A in mix_options_str.split(',')if A.strip()];base_dir=os.path.dirname(os.path.abspath(__file__));output_dir=os.path.join(base_dir,'processed');temp_dir=os.path.join(base_dir,'temp',job_id)
	for d in[output_dir,temp_dir]:os.makedirs(d,exist_ok=_A)
	final_output=os.path.join(output_dir,f"{job_id}_final.mp3")if FFMPEG_PATH else os.path.join(output_dir,f"{job_id}_final.wav");print(f"=== Music Extractor ===",flush=_A);print(f"Job: {job_id} | Inst: {instrument} | Effects: {len(effects_opts)} settings",flush=_A)
	try:
		if os.path.splitext(input_file)[1].lower()!=_F:work_file=os.path.join(temp_dir,'input_converted.wav');convert_to_wav(input_file,work_file)
		else:work_file=input_file
		if instrument=='mix_only':y,sr=librosa.load(work_file,sr=22050,mono=_A);sf.write(final_output.replace(_G,_F),y,sr);print('Done (Mix Only).',flush=_A)
		else:
			vocals_path,accompaniment_path,other_path=separate_with_demucs(work_file,temp_dir);instrument_audio=_D;inst_sr=22050
			if instrument=='split_flute':print("  [Extraction] Isolating Flute from 'Other' stem... (Bandpass 260-8000Hz)",flush=_A);other,sr=librosa.load(other_path,sr=22050,mono=_A);sos=scipy.signal.butter(4,[260,8000],btype='bandpass',fs=sr,output=_E);instrument_audio=scipy.signal.sosfilt(sos,other);instrument_audio=normalize_audio(instrument_audio,_C);inst_sr=sr
			elif _J in mix_options_list:instrument_audio,inst_sr=synthesize_instrument(vocals_path,instrument,effects_opts)
			mix_val=effects_opts.get('mix_balance',.5)
			try:mix_val=float(mix_val)
			except:mix_val=.5
			dynamic_mix(vocals_path,accompaniment_path,instrument_audio,inst_sr,mix_options_list,final_output,mix_val);print(f"Output: {final_output}",flush=_A)
	except Exception as e:
		import traceback;traceback.print_exc();status_dir=os.path.join(base_dir,_K);os.makedirs(status_dir,exist_ok=_A)
		with open(os.path.join(status_dir,f"{job_id}.json"),'w')as f:json.dump({_K:'error','message':str(e)},f)
		sys.exit(1)